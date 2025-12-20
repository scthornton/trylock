"""
TRYLOCK Security Gateway

Request/response filtering gateway that sits in front of LLM endpoints.
Provides pre-flight attack detection and post-flight response validation.

Architecture:
    Client → Gateway → Sidecar Check → LLM → Response Filter → Client

Features:
- Pre-flight: Classify incoming requests before LLM processing
- Post-flight: Validate responses before returning to client
- Rate limiting and abuse detection
- Audit logging for security analysis
- Configurable intervention policies

Usage:
    # As middleware
    from deployment.gateway import TRYLOCKGateway
    gateway = TRYLOCKGateway(config)

    # FastAPI integration
    app = FastAPI()
    gateway.mount(app)

    # Standalone proxy
    python -m deployment.gateway --port 8080 --target http://localhost:11434
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal
from collections import defaultdict

try:
    from fastapi import FastAPI, Request, Response, HTTPException
    from fastapi.middleware.base import BaseHTTPMiddleware
    from fastapi.responses import JSONResponse
    import httpx
except ImportError:
    FastAPI = None
    Request = None
    httpx = None

import yaml


class ThreatLevel(Enum):
    """Threat classification levels."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InterventionAction(Enum):
    """Actions to take based on threat assessment."""
    ALLOW = "allow"              # Process normally
    WARN = "warn"                # Log warning, allow
    ENHANCE = "enhance"          # Enable enhanced defenses
    MODIFY = "modify"            # Modify request/response
    BLOCK = "block"              # Block request entirely
    CHALLENGE = "challenge"      # Require additional verification


@dataclass
class GatewayConfig:
    """Configuration for the security gateway."""

    # Sidecar connection
    sidecar_url: str = "http://localhost:8081"
    sidecar_timeout: float = 1.0  # seconds
    sidecar_enabled: bool = True

    # Threat thresholds
    warn_threshold: float = 0.3
    block_threshold: float = 0.8

    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # Abuse detection
    abuse_detection_enabled: bool = True
    max_prompt_length: int = 32000
    max_turns_per_session: int = 50

    # Response filtering
    filter_responses: bool = True
    max_response_length: int = 64000

    # Logging
    audit_logging: bool = True
    log_file: str = "gateway_audit.jsonl"
    log_level: str = "INFO"

    # Intervention policies by threat level
    policies: dict[str, str] = field(default_factory=lambda: {
        "safe": "allow",
        "low": "allow",
        "medium": "warn",
        "high": "enhance",
        "critical": "block",
    })

    # Target LLM endpoint
    target_url: str = "http://localhost:11434/api/chat"
    target_timeout: float = 120.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GatewayConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("gateway", data))


@dataclass
class RequestContext:
    """Context for a single request through the gateway."""

    request_id: str
    timestamp: datetime
    client_ip: str
    session_id: str | None

    # Request data
    messages: list[dict]
    model: str | None
    raw_body: bytes

    # Assessment
    threat_level: ThreatLevel = ThreatLevel.SAFE
    risk_score: float = 0.0
    sidecar_classification: str | None = None

    # Intervention
    action: InterventionAction = InterventionAction.ALLOW
    modifications: list[str] = field(default_factory=list)

    # Response
    response_body: bytes | None = None
    response_filtered: bool = False

    # Timing
    sidecar_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        cutoff = now - self.window_seconds

        # Clean old requests
        self.requests[key] = [t for t in self.requests[key] if t > cutoff]

        if len(self.requests[key]) >= self.max_requests:
            return False

        self.requests[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in window."""
        now = time.time()
        cutoff = now - self.window_seconds
        self.requests[key] = [t for t in self.requests[key] if t > cutoff]
        return max(0, self.max_requests - len(self.requests[key]))


class AbuseDetector:
    """Detect potential abuse patterns."""

    def __init__(self, config: GatewayConfig):
        self.config = config
        self.session_turns: dict[str, int] = defaultdict(int)
        self.flagged_patterns: list[str] = [
            # Repeated attack patterns
            r"ignore previous",
            r"disregard.*instructions",
            r"you are now",
            r"new persona",
            r"jailbreak",
            r"DAN mode",
        ]

    def check(self, ctx: RequestContext) -> list[str]:
        """Check for abuse patterns. Returns list of warnings."""
        warnings = []

        # Check prompt length
        total_length = sum(
            len(m.get("content", "")) for m in ctx.messages
        )
        if total_length > self.config.max_prompt_length:
            warnings.append(f"Prompt length ({total_length}) exceeds limit")

        # Check session turns
        if ctx.session_id:
            self.session_turns[ctx.session_id] += 1
            if self.session_turns[ctx.session_id] > self.config.max_turns_per_session:
                warnings.append("Session turn limit exceeded")

        # Check for known attack patterns
        import re
        content = " ".join(m.get("content", "") for m in ctx.messages).lower()
        for pattern in self.flagged_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                warnings.append(f"Flagged pattern detected: {pattern}")

        return warnings


class AuditLogger:
    """Audit logging for security analysis."""

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(self, ctx: RequestContext, event: str, details: dict | None = None):
        """Log an audit event."""
        entry = {
            "timestamp": ctx.timestamp.isoformat(),
            "request_id": ctx.request_id,
            "client_ip": ctx.client_ip,
            "session_id": ctx.session_id,
            "event": event,
            "threat_level": ctx.threat_level.value,
            "risk_score": ctx.risk_score,
            "action": ctx.action.value,
            "sidecar_latency_ms": ctx.sidecar_latency_ms,
            "llm_latency_ms": ctx.llm_latency_ms,
            "total_latency_ms": ctx.total_latency_ms,
        }
        if details:
            entry["details"] = details

        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


class SidecarClient:
    """Client for communicating with sidecar classifier."""

    def __init__(self, base_url: str, timeout: float = 1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def classify(self, messages: list[dict]) -> dict:
        """Classify conversation via sidecar."""
        if httpx is None:
            raise ImportError("httpx required for async requests")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/classify",
                json={"messages": messages},
            )
            response.raise_for_status()
            return response.json()

    def classify_sync(self, messages: list[dict]) -> dict:
        """Synchronous classification."""
        if httpx is None:
            raise ImportError("httpx required")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/classify",
                json={"messages": messages},
            )
            response.raise_for_status()
            return response.json()


class ResponseFilter:
    """Filter and validate LLM responses."""

    def __init__(self, config: GatewayConfig):
        self.config = config

        # Patterns that should trigger filtering
        self.dangerous_patterns = [
            r"here(?:'s| is) (?:how to|the code|a script)",
            r"step \d+:.*(?:hack|exploit|attack)",
            r"```(?:python|bash|sh).*(?:os\.system|subprocess|eval\()",
        ]

    def filter(self, response_text: str, ctx: RequestContext) -> tuple[str, bool]:
        """
        Filter response if needed.

        Returns:
            Tuple of (filtered_text, was_modified)
        """
        import re

        was_modified = False

        # Length check
        if len(response_text) > self.config.max_response_length:
            response_text = response_text[:self.config.max_response_length]
            response_text += "\n\n[Response truncated for safety]"
            was_modified = True

        # Pattern check (only if high threat)
        if ctx.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            for pattern in self.dangerous_patterns:
                if re.search(pattern, response_text, re.IGNORECASE | re.DOTALL):
                    # Log but don't modify - let intervention handler decide
                    ctx.modifications.append(f"Dangerous pattern detected: {pattern}")

        return response_text, was_modified


class TRYLOCKGateway:
    """
    Main security gateway class.

    Coordinates all security components for request/response filtering.
    """

    def __init__(self, config: GatewayConfig | None = None):
        self.config = config or GatewayConfig()

        # Initialize components
        self.rate_limiter = RateLimiter(
            self.config.rate_limit_requests,
            self.config.rate_limit_window,
        )
        self.abuse_detector = AbuseDetector(self.config)
        self.audit_logger = AuditLogger(self.config.log_file) if self.config.audit_logging else None
        self.sidecar = SidecarClient(self.config.sidecar_url, self.config.sidecar_timeout)
        self.response_filter = ResponseFilter(self.config)

        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger("trylock.gateway")

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())[:8]

    def _extract_messages(self, body: dict) -> list[dict]:
        """Extract messages from request body."""
        # Handle different API formats
        if "messages" in body:
            return body["messages"]
        elif "prompt" in body:
            return [{"role": "user", "content": body["prompt"]}]
        elif "conversation" in body:
            return body["conversation"]
        return []

    def _determine_threat_level(self, risk_score: float) -> ThreatLevel:
        """Map risk score to threat level."""
        if risk_score >= 0.8:
            return ThreatLevel.CRITICAL
        elif risk_score >= 0.6:
            return ThreatLevel.HIGH
        elif risk_score >= 0.4:
            return ThreatLevel.MEDIUM
        elif risk_score >= 0.2:
            return ThreatLevel.LOW
        return ThreatLevel.SAFE

    def _determine_action(self, threat_level: ThreatLevel) -> InterventionAction:
        """Determine intervention action based on threat level."""
        action_str = self.config.policies.get(
            threat_level.value, "allow"
        )
        return InterventionAction(action_str)

    async def assess_request(self, ctx: RequestContext) -> RequestContext:
        """
        Assess request threat level.

        Updates ctx with threat assessment.
        """
        # Check abuse patterns
        abuse_warnings = self.abuse_detector.check(ctx)
        if abuse_warnings:
            ctx.risk_score = max(ctx.risk_score, 0.5)
            ctx.modifications.extend(abuse_warnings)

        # Sidecar classification
        if self.config.sidecar_enabled:
            try:
                start = time.time()
                result = await self.sidecar.classify(ctx.messages)
                ctx.sidecar_latency_ms = (time.time() - start) * 1000

                ctx.sidecar_classification = result.get("classification")
                ctx.risk_score = max(ctx.risk_score, result.get("risk_score", 0.0))

            except Exception as e:
                self.logger.warning(f"Sidecar error: {e}")
                # Continue without sidecar on error

        # Determine threat level and action
        ctx.threat_level = self._determine_threat_level(ctx.risk_score)
        ctx.action = self._determine_action(ctx.threat_level)

        return ctx

    async def process_request(
        self,
        body: dict,
        client_ip: str = "unknown",
        session_id: str | None = None,
    ) -> tuple[dict | None, RequestContext]:
        """
        Process a request through the gateway.

        Returns:
            Tuple of (response_dict, context)
            response_dict is None if blocked
        """
        start_time = time.time()

        # Create context
        ctx = RequestContext(
            request_id=self._generate_request_id(),
            timestamp=datetime.utcnow(),
            client_ip=client_ip,
            session_id=session_id,
            messages=self._extract_messages(body),
            model=body.get("model"),
            raw_body=json.dumps(body).encode(),
        )

        # Rate limiting
        if self.config.rate_limit_enabled:
            if not self.rate_limiter.is_allowed(client_ip):
                ctx.action = InterventionAction.BLOCK
                ctx.modifications.append("Rate limit exceeded")
                if self.audit_logger:
                    self.audit_logger.log(ctx, "rate_limited")
                return None, ctx

        # Assess threat
        ctx = await self.assess_request(ctx)

        # Log assessment
        if self.audit_logger:
            self.audit_logger.log(ctx, "assessed", {
                "sidecar_classification": ctx.sidecar_classification,
            })

        # Handle block action
        if ctx.action == InterventionAction.BLOCK:
            if self.audit_logger:
                self.audit_logger.log(ctx, "blocked")
            return None, ctx

        # Forward to LLM
        try:
            llm_start = time.time()

            async with httpx.AsyncClient(timeout=self.config.target_timeout) as client:
                response = await client.post(
                    self.config.target_url,
                    json=body,
                )
                response.raise_for_status()
                llm_response = response.json()

            ctx.llm_latency_ms = (time.time() - llm_start) * 1000

        except Exception as e:
            self.logger.error(f"LLM error: {e}")
            ctx.modifications.append(f"LLM error: {str(e)}")
            if self.audit_logger:
                self.audit_logger.log(ctx, "llm_error", {"error": str(e)})
            raise

        # Filter response
        if self.config.filter_responses:
            response_text = self._extract_response_text(llm_response)
            filtered_text, was_modified = self.response_filter.filter(response_text, ctx)

            if was_modified:
                ctx.response_filtered = True
                llm_response = self._update_response_text(llm_response, filtered_text)

        ctx.total_latency_ms = (time.time() - start_time) * 1000

        # Final logging
        if self.audit_logger:
            self.audit_logger.log(ctx, "completed")

        return llm_response, ctx

    def _extract_response_text(self, response: dict) -> str:
        """Extract text from LLM response."""
        # Handle different formats
        if "message" in response:
            return response["message"].get("content", "")
        elif "choices" in response:
            return response["choices"][0].get("message", {}).get("content", "")
        elif "content" in response:
            return response["content"]
        return str(response)

    def _update_response_text(self, response: dict, new_text: str) -> dict:
        """Update response with filtered text."""
        if "message" in response:
            response["message"]["content"] = new_text
        elif "choices" in response:
            response["choices"][0]["message"]["content"] = new_text
        elif "content" in response:
            response["content"] = new_text
        return response

    def create_fastapi_app(self) -> "FastAPI":
        """Create FastAPI application with gateway middleware."""
        if FastAPI is None:
            raise ImportError("fastapi required")

        app = FastAPI(title="TRYLOCK Security Gateway")
        gateway = self

        @app.post("/v1/chat/completions")
        @app.post("/api/chat")
        async def chat_endpoint(request: Request):
            body = await request.json()
            client_ip = request.client.host if request.client else "unknown"
            session_id = request.headers.get("X-Session-ID")

            try:
                response, ctx = await gateway.process_request(
                    body, client_ip, session_id
                )

                if response is None:
                    raise HTTPException(
                        status_code=429 if "Rate limit" in str(ctx.modifications) else 403,
                        detail={
                            "error": "Request blocked",
                            "reason": ctx.modifications,
                            "request_id": ctx.request_id,
                        }
                    )

                return JSONResponse(
                    content=response,
                    headers={
                        "X-Request-ID": ctx.request_id,
                        "X-Threat-Level": ctx.threat_level.value,
                        "X-Gateway-Latency-Ms": str(int(ctx.total_latency_ms)),
                    }
                )

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health():
            return {"status": "healthy", "sidecar_enabled": gateway.config.sidecar_enabled}

        @app.get("/stats")
        async def stats():
            return {
                "rate_limiter_keys": len(gateway.rate_limiter.requests),
                "session_count": len(gateway.abuse_detector.session_turns),
            }

        return app


def create_gateway(config_path: str | None = None) -> TRYLOCKGateway:
    """Create gateway instance from config."""
    if config_path:
        config = GatewayConfig.from_yaml(config_path)
    else:
        config = GatewayConfig()

    return TRYLOCKGateway(config)


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="TRYLOCK Security Gateway")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--target", type=str, help="Target LLM URL")
    parser.add_argument("--sidecar", type=str, help="Sidecar URL")

    args = parser.parse_args()

    if args.config:
        config = GatewayConfig.from_yaml(args.config)
    else:
        config = GatewayConfig()

    if args.target:
        config.target_url = args.target
    if args.sidecar:
        config.sidecar_url = args.sidecar

    gateway = TRYLOCKGateway(config)
    app = gateway.create_fastapi_app()

    print(f"Starting TRYLOCK Gateway on {args.host}:{args.port}")
    print(f"Target LLM: {config.target_url}")
    print(f"Sidecar: {config.sidecar_url}")

    uvicorn.run(app, host=args.host, port=args.port)
