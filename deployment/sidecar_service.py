"""
TRYLOCK Sidecar Inference Service

Standalone service for running the sidecar classifier.
Provides HTTP/gRPC endpoints for real-time conversation classification.

Features:
- Low-latency classification (<100ms target)
- Batched inference for throughput
- Health monitoring and metrics
- Model hot-reloading

Usage:
    # Start service
    python -m deployment.sidecar_service --model ./outputs/trylock-sidecar --port 8081

    # Docker
    docker run -p 8081:8081 trylock-sidecar:latest

    # Query
    curl -X POST http://localhost:8081/classify \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "Hello"}]}'
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Thread, Lock

import yaml

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    FastAPI = None
    BaseModel = object
    uvicorn = None

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


# Request/Response models

class Message(BaseModel if BaseModel != object else object):
    """Single conversation message."""
    role: str = "user"
    content: str = ""


class ClassifyRequest(BaseModel if BaseModel != object else object):
    """Classification request."""
    messages: list[dict] = Field(default_factory=list)
    session_id: str | None = None
    include_history: bool = False


class ClassifyResponse(BaseModel if BaseModel != object else object):
    """Classification response."""
    classification: str  # SAFE, WARN, ATTACK
    class_id: int
    probabilities: dict[str, float]
    risk_score: float
    action: str
    latency_ms: float


class BatchClassifyRequest(BaseModel if BaseModel != object else object):
    """Batch classification request."""
    conversations: list[list[dict]]


class BatchClassifyResponse(BaseModel if BaseModel != object else object):
    """Batch classification response."""
    results: list[ClassifyResponse]
    total_latency_ms: float


class HealthResponse(BaseModel if BaseModel != object else object):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


class MetricsResponse(BaseModel if BaseModel != object else object):
    """Metrics response."""
    total_requests: int
    total_classifications: int
    average_latency_ms: float
    classifications_by_class: dict[str, int]
    error_count: int


@dataclass
class SidecarServiceConfig:
    """Configuration for sidecar service."""

    # Model
    model_path: str = "./outputs/trylock-sidecar"
    device: str = "auto"
    max_seq_length: int = 2048

    # Classification
    label_names: list[str] = field(default_factory=lambda: ["SAFE", "WARN", "ATTACK"])
    warn_threshold: float = 0.3
    attack_threshold: float = 0.7

    # Server
    host: str = "0.0.0.0"
    port: int = 8081
    workers: int = 1

    # Batching
    batch_enabled: bool = True
    batch_size: int = 8
    batch_timeout_ms: float = 10.0

    # Performance
    use_half_precision: bool = True
    compile_model: bool = False  # torch.compile

    # Caching
    cache_enabled: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 300

    # Logging
    log_level: str = "INFO"
    log_requests: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SidecarServiceConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("sidecar_service", data))


class ClassificationCache:
    """LRU cache for classification results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: dict[str, tuple[dict, float]] = {}
        self.lock = Lock()

    def _hash_messages(self, messages: list[dict]) -> str:
        """Create hash key for messages."""
        import hashlib
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, messages: list[dict]) -> dict | None:
        """Get cached result if available and not expired."""
        key = self._hash_messages(messages)

        with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return result
                else:
                    del self.cache[key]
        return None

    def set(self, messages: list[dict], result: dict):
        """Cache a result."""
        key = self._hash_messages(messages)

        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            self.cache[key] = (result, time.time())

    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()


class BatchProcessor:
    """
    Batch processor for efficient inference.

    Collects requests and processes in batches for better GPU utilization.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: SidecarServiceConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.queue: Queue = Queue()
        self.results: dict[int, asyncio.Future] = {}
        self.request_id = 0
        self.lock = Lock()

        # Start background processor
        self.running = True
        self.processor_thread = Thread(target=self._process_loop, daemon=True)
        self.processor_thread.start()

    def _process_loop(self):
        """Background processing loop."""
        while self.running:
            batch = []
            request_ids = []

            # Collect batch
            try:
                # Wait for first item
                item = self.queue.get(timeout=0.1)
                batch.append(item[1])
                request_ids.append(item[0])

                # Collect more items up to batch size
                deadline = time.time() + self.config.batch_timeout_ms / 1000
                while len(batch) < self.config.batch_size and time.time() < deadline:
                    try:
                        item = self.queue.get_nowait()
                        batch.append(item[1])
                        request_ids.append(item[0])
                    except Empty:
                        time.sleep(0.001)
                        break

            except Empty:
                continue

            # Process batch
            if batch:
                results = self._process_batch(batch)

                # Return results
                for req_id, result in zip(request_ids, results):
                    if req_id in self.results:
                        future = self.results.pop(req_id)
                        if not future.done():
                            future.get_loop().call_soon_threadsafe(
                                future.set_result, result
                            )

    def _process_batch(self, messages_batch: list[list[dict]]) -> list[dict]:
        """Process a batch of conversations."""
        # Format conversations
        texts = [self._format_conversation(msgs) for msgs in messages_batch]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )

        if torch is not None:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Process outputs
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for i, prob in enumerate(probs):
            predicted_class = int(prob.argmax())
            risk_score = prob[1] * 0.3 + prob[2] * 0.7

            results.append({
                "classification": self.config.label_names[predicted_class],
                "class_id": predicted_class,
                "probabilities": {
                    name: float(p) for name, p in zip(self.config.label_names, prob)
                },
                "risk_score": float(risk_score),
                "action": self._get_action(predicted_class, risk_score),
            })

        return results

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format conversation for model input."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<{role}>\n{content}\n</{role}>")
        return "\n".join(parts)

    def _get_action(self, class_id: int, risk_score: float) -> str:
        """Get recommended action."""
        if class_id == 0 and risk_score < self.config.warn_threshold:
            return "CONTINUE"
        elif class_id == 1 or self.config.warn_threshold <= risk_score < self.config.attack_threshold:
            return "ENHANCE_MONITORING"
        else:
            return "INTERVENE"

    async def classify(self, messages: list[dict]) -> dict:
        """Submit classification request."""
        with self.lock:
            self.request_id += 1
            req_id = self.request_id

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        self.results[req_id] = future
        self.queue.put((req_id, messages))

        return await future

    def shutdown(self):
        """Shutdown processor."""
        self.running = False
        self.processor_thread.join(timeout=1.0)


class SidecarClassifierService:
    """
    Main sidecar classifier service.
    """

    def __init__(self, config: SidecarServiceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.batch_processor = None
        self.cache = None
        self.start_time = time.time()

        # Metrics
        self.total_requests = 0
        self.total_classifications = 0
        self.total_latency_ms = 0.0
        self.classifications_by_class = {name: 0 for name in config.label_names}
        self.error_count = 0

        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger("trylock.sidecar")

    def load_model(self):
        """Load the classification model."""
        if AutoModelForSequenceClassification is None:
            raise ImportError("transformers required")

        self.logger.info(f"Loading model from {self.config.model_path}")

        # Determine device
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        dtype = torch.float16 if self.config.use_half_precision and device != "cpu" else torch.float32

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
        ).to(device)

        self.model.eval()

        # Compile if requested
        if self.config.compile_model and hasattr(torch, "compile"):
            self.logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)

        # Load thresholds from model directory
        thresholds_path = Path(self.config.model_path) / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                thresholds = json.load(f)
            self.config.warn_threshold = thresholds.get("warn_threshold", self.config.warn_threshold)
            self.config.attack_threshold = thresholds.get("attack_threshold", self.config.attack_threshold)
            self.config.label_names = thresholds.get("label_names", self.config.label_names)

        # Initialize batch processor
        if self.config.batch_enabled:
            self.batch_processor = BatchProcessor(self.model, self.tokenizer, self.config)

        # Initialize cache
        if self.config.cache_enabled:
            self.cache = ClassificationCache(
                max_size=self.config.cache_size,
                ttl_seconds=self.config.cache_ttl_seconds,
            )

        self.logger.info(f"Model loaded on {device}")

    def _format_conversation(self, messages: list[dict]) -> str:
        """Format conversation for model input."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<{role}>\n{content}\n</{role}>")
        return "\n".join(parts)

    def _classify_single(self, messages: list[dict]) -> dict:
        """Classify a single conversation (non-batched)."""
        text = self._format_conversation(messages)

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class = int(probs.argmax())
        risk_score = probs[1] * 0.3 + probs[2] * 0.7

        return {
            "classification": self.config.label_names[predicted_class],
            "class_id": predicted_class,
            "probabilities": {
                name: float(p) for name, p in zip(self.config.label_names, probs)
            },
            "risk_score": float(risk_score),
            "action": self._get_action(predicted_class, risk_score),
        }

    def _get_action(self, class_id: int, risk_score: float) -> str:
        """Get recommended action."""
        if class_id == 0 and risk_score < self.config.warn_threshold:
            return "CONTINUE"
        elif class_id == 1 or self.config.warn_threshold <= risk_score < self.config.attack_threshold:
            return "ENHANCE_MONITORING"
        else:
            return "INTERVENE"

    async def classify(self, messages: list[dict]) -> dict:
        """Classify a conversation."""
        start_time = time.time()

        # Check cache
        if self.cache:
            cached = self.cache.get(messages)
            if cached:
                return {**cached, "cached": True, "latency_ms": 0.0}

        # Classify
        if self.batch_processor:
            result = await self.batch_processor.classify(messages)
        else:
            result = self._classify_single(messages)

        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = latency_ms

        # Update cache
        if self.cache:
            self.cache.set(messages, result)

        # Update metrics
        self.total_classifications += 1
        self.total_latency_ms += latency_ms
        self.classifications_by_class[result["classification"]] += 1

        return result

    def get_metrics(self) -> dict:
        """Get service metrics."""
        avg_latency = (
            self.total_latency_ms / self.total_classifications
            if self.total_classifications > 0 else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "total_classifications": self.total_classifications,
            "average_latency_ms": avg_latency,
            "classifications_by_class": self.classifications_by_class.copy(),
            "error_count": self.error_count,
            "cache_size": len(self.cache.cache) if self.cache else 0,
        }

    def get_health(self) -> dict:
        """Get health status."""
        return {
            "status": "healthy" if self.model is not None else "unhealthy",
            "model_loaded": self.model is not None,
            "device": str(self.model.device) if self.model else "none",
            "uptime_seconds": time.time() - self.start_time,
        }

    def create_app(self) -> "FastAPI":
        """Create FastAPI application."""
        if FastAPI is None:
            raise ImportError("fastapi required")

        app = FastAPI(
            title="TRYLOCK Sidecar Classifier",
            description="Real-time conversation security classification",
            version="1.0.0",
        )

        service = self

        @app.on_event("startup")
        async def startup():
            service.load_model()

        @app.on_event("shutdown")
        async def shutdown():
            if service.batch_processor:
                service.batch_processor.shutdown()

        @app.post("/classify", response_model=ClassifyResponse)
        async def classify(request: ClassifyRequest):
            service.total_requests += 1

            if service.config.log_requests:
                service.logger.info(f"Classify request: {len(request.messages)} messages")

            try:
                result = await service.classify(request.messages)
                return ClassifyResponse(**result)
            except Exception as e:
                service.error_count += 1
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/classify/batch", response_model=BatchClassifyResponse)
        async def classify_batch(request: BatchClassifyRequest):
            service.total_requests += 1
            start_time = time.time()

            try:
                results = []
                for conversation in request.conversations:
                    result = await service.classify(conversation)
                    results.append(ClassifyResponse(**result))

                total_latency = (time.time() - start_time) * 1000

                return BatchClassifyResponse(
                    results=results,
                    total_latency_ms=total_latency,
                )
            except Exception as e:
                service.error_count += 1
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health", response_model=HealthResponse)
        async def health():
            return HealthResponse(**service.get_health())

        @app.get("/metrics", response_model=MetricsResponse)
        async def metrics():
            return MetricsResponse(**service.get_metrics())

        @app.post("/cache/clear")
        async def clear_cache():
            if service.cache:
                service.cache.clear()
            return {"status": "cache cleared"}

        @app.get("/")
        async def root():
            return {
                "service": "TRYLOCK Sidecar Classifier",
                "version": "1.0.0",
                "endpoints": [
                    "/classify",
                    "/classify/batch",
                    "/health",
                    "/metrics",
                ],
            }

        return app

    def run(self):
        """Run the service."""
        if uvicorn is None:
            raise ImportError("uvicorn required")

        app = self.create_app()
        uvicorn.run(
            app,
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
        )


def create_service(
    model_path: str | None = None,
    config_path: str | None = None,
) -> SidecarClassifierService:
    """Create sidecar service."""
    if config_path:
        config = SidecarServiceConfig.from_yaml(config_path)
    else:
        config = SidecarServiceConfig()

    if model_path:
        config.model_path = model_path

    return SidecarClassifierService(config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRYLOCK Sidecar Classifier Service")
    parser.add_argument("--model", type=str, default="./outputs/trylock-sidecar", help="Model path")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--port", type=int, default=8081, help="Port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--no-batch", action="store_true", help="Disable batching")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    if args.config:
        config = SidecarServiceConfig.from_yaml(args.config)
    else:
        config = SidecarServiceConfig()

    config.model_path = args.model
    config.port = args.port
    config.host = args.host
    config.device = args.device
    config.batch_enabled = not args.no_batch
    config.cache_enabled = not args.no_cache

    print(f"Starting TRYLOCK Sidecar Service")
    print(f"  Model: {config.model_path}")
    print(f"  Port: {config.port}")
    print(f"  Device: {config.device}")
    print(f"  Batching: {config.batch_enabled}")
    print(f"  Caching: {config.cache_enabled}")

    service = SidecarClassifierService(config)
    service.run()
