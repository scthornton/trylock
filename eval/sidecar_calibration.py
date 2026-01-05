#!/usr/bin/env python3
"""
TRYLOCK Sidecar Calibration Module

Implements:
1. Temperature scaling for calibrated probabilities
2. Risk scoring (weighted probability combination)
3. ROC and Precision-Recall curve generation
4. Calibration curve analysis (reliability diagrams)
5. Cost-weighted evaluation metrics

Usage:
    python sidecar_calibration.py --sidecar-path path/to/sidecar --data-path path/to/test.json
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError("PyTorch required: pip install torch")

try:
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, average_precision_score,
        brier_score_loss, log_loss
    )
    from sklearn.calibration import calibration_curve
except ImportError:
    raise ImportError("scikit-learn required: pip install scikit-learn")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    warnings.warn("matplotlib not available, plotting disabled")


@dataclass
class CalibrationResult:
    """Results from calibration analysis."""
    # Temperature scaling
    optimal_temperature: float

    # Before calibration
    ece_before: float  # Expected Calibration Error
    brier_before: float
    log_loss_before: float

    # After calibration
    ece_after: float
    brier_after: float
    log_loss_after: float

    # ROC metrics
    roc_auc_attack: float
    roc_auc_safe: float

    # PR metrics
    pr_auc_attack: float
    pr_auc_safe: float

    # Cost-weighted metrics
    cost_weighted_error: float

    def to_dict(self) -> dict:
        return {
            "optimal_temperature": self.optimal_temperature,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "brier_before": self.brier_before,
            "brier_after": self.brier_after,
            "log_loss_before": self.log_loss_before,
            "log_loss_after": self.log_loss_after,
            "roc_auc_attack": self.roc_auc_attack,
            "roc_auc_safe": self.roc_auc_safe,
            "pr_auc_attack": self.pr_auc_attack,
            "pr_auc_safe": self.pr_auc_safe,
            "cost_weighted_error": self.cost_weighted_error,
        }


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for neural network calibration.

    Scales logits by a learned temperature parameter T:
    p(y|x) = softmax(z/T)

    Reference: Guo et al., "On Calibration of Modern Neural Networks"
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Learn optimal temperature on validation set.

        Args:
            logits: Raw logits from sidecar [N, 3]
            labels: True labels [N]
            lr: Learning rate
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature value
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.temperature.item()


class RiskScorer:
    """
    Weighted risk scoring for sidecar outputs.

    Combines class probabilities into a single risk score:
    risk = w_attack * p(ATTACK) + w_warn * p(WARN) + w_safe * p(SAFE)

    Default weights prioritize attack detection.
    """

    def __init__(
        self,
        w_attack: float = 1.0,
        w_warn: float = 0.5,
        w_safe: float = 0.0,
    ):
        self.w_attack = w_attack
        self.w_warn = w_warn
        self.w_safe = w_safe

    def score(self, probs: np.ndarray) -> np.ndarray:
        """
        Compute risk scores from probabilities.

        Args:
            probs: Probability matrix [N, 3] with columns [SAFE, WARN, ATTACK]

        Returns:
            Risk scores [N] in range [0, 1]
        """
        scores = (
            self.w_safe * probs[:, 0] +
            self.w_warn * probs[:, 1] +
            self.w_attack * probs[:, 2]
        )
        # Normalize to [0, 1]
        return scores / (self.w_safe + self.w_warn + self.w_attack)


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average gap between predicted confidence and accuracy.

    Args:
        probs: Predicted probabilities [N, C] or [N] for binary
        labels: True labels [N]
        n_bins: Number of confidence bins

    Returns:
        ECE value (lower is better)
    """
    if probs.ndim == 2:
        # Multi-class: use max probability
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs > 0.5).astype(int)

    accuracies = (predictions == labels)

    # Bin samples by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * np.abs(avg_confidence - avg_accuracy)

    return ece


def compute_cost_weighted_error(
    predictions: np.ndarray,
    labels: np.ndarray,
    cost_matrix: Optional[np.ndarray] = None,
) -> float:
    """
    Compute cost-weighted classification error.

    Default cost matrix heavily penalizes ATTACK→SAFE misclassifications.

    Args:
        predictions: Predicted class labels [N]
        labels: True labels [N]
        cost_matrix: [3, 3] matrix where C[i,j] = cost of predicting j when true is i

    Returns:
        Mean cost-weighted error
    """
    if cost_matrix is None:
        # Default: heavily penalize missed attacks
        # Rows = true class, Cols = predicted class
        # Classes: 0=SAFE, 1=WARN, 2=ATTACK
        cost_matrix = np.array([
            [0.0, 0.5, 1.0],   # True SAFE: predicting ATTACK costs 1.0
            [0.5, 0.0, 0.5],   # True WARN: moderate costs either way
            [5.0, 2.0, 0.0],   # True ATTACK: missing it (pred SAFE) costs 5.0!
        ])

    total_cost = 0.0
    for true_label, pred_label in zip(labels, predictions):
        total_cost += cost_matrix[int(true_label), int(pred_label)]

    return total_cost / len(labels)


def generate_roc_curves(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> dict:
    """
    Generate ROC curves for each class.

    Args:
        probs: Probability matrix [N, 3]
        labels: True labels [N]
        save_path: Optional path to save figure

    Returns:
        Dict with FPR, TPR, and AUC for each class
    """
    class_names = ["SAFE", "WARN", "ATTACK"]
    results = {}

    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, name in enumerate(class_names):
        # Binary labels for this class
        binary_labels = (labels == i).astype(int)
        class_probs = probs[:, i]

        fpr, tpr, thresholds = roc_curve(binary_labels, class_probs)
        roc_auc = auc(fpr, tpr)

        results[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": roc_auc,
        }

        if plt is not None:
            axes[i].plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {roc_auc:.3f}')
            axes[i].plot([0, 1], [0, 1], 'k--', lw=1)
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'ROC Curve - {name}')
            axes[i].legend(loc='lower right')
            axes[i].set_xlim([0, 1])
            axes[i].set_ylim([0, 1])

    if plt is not None and save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return results


def generate_pr_curves(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> dict:
    """
    Generate Precision-Recall curves for each class.

    Args:
        probs: Probability matrix [N, 3]
        labels: True labels [N]
        save_path: Optional path to save figure

    Returns:
        Dict with precision, recall, and AP for each class
    """
    class_names = ["SAFE", "WARN", "ATTACK"]
    results = {}

    if plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, name in enumerate(class_names):
        binary_labels = (labels == i).astype(int)
        class_probs = probs[:, i]

        precision, recall, thresholds = precision_recall_curve(binary_labels, class_probs)
        ap = average_precision_score(binary_labels, class_probs)

        results[name] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "average_precision": ap,
        }

        if plt is not None:
            axes[i].plot(recall, precision, 'b-', lw=2, label=f'AP = {ap:.3f}')
            axes[i].set_xlabel('Recall')
            axes[i].set_ylabel('Precision')
            axes[i].set_title(f'Precision-Recall Curve - {name}')
            axes[i].legend(loc='lower left')
            axes[i].set_xlim([0, 1])
            axes[i].set_ylim([0, 1])

    if plt is not None and save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return results


def generate_calibration_plot(
    probs: np.ndarray,
    labels: np.ndarray,
    probs_calibrated: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    n_bins: int = 10,
) -> dict:
    """
    Generate calibration (reliability) diagram.

    Args:
        probs: Uncalibrated probabilities [N, 3]
        labels: True labels [N]
        probs_calibrated: Optional calibrated probabilities
        save_path: Optional path to save figure
        n_bins: Number of calibration bins

    Returns:
        Calibration curve data
    """
    if plt is None:
        return {}

    class_names = ["SAFE", "WARN", "ATTACK"]
    results = {}

    n_rows = 1 if probs_calibrated is None else 2
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(class_names):
        binary_labels = (labels == i).astype(int)

        # Uncalibrated
        prob_true, prob_pred = calibration_curve(
            binary_labels, probs[:, i], n_bins=n_bins, strategy='uniform'
        )

        results[f"{name}_uncalibrated"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }

        axes[0, i].plot(prob_pred, prob_true, 'b-o', label='Uncalibrated')
        axes[0, i].plot([0, 1], [0, 1], 'k--', label='Perfect')
        axes[0, i].set_xlabel('Mean Predicted Probability')
        axes[0, i].set_ylabel('Fraction of Positives')
        axes[0, i].set_title(f'Calibration - {name}')
        axes[0, i].legend()

        # Calibrated (if provided)
        if probs_calibrated is not None:
            prob_true_cal, prob_pred_cal = calibration_curve(
                binary_labels, probs_calibrated[:, i], n_bins=n_bins, strategy='uniform'
            )

            results[f"{name}_calibrated"] = {
                "prob_true": prob_true_cal.tolist(),
                "prob_pred": prob_pred_cal.tolist(),
            }

            axes[1, i].plot(prob_pred_cal, prob_true_cal, 'g-o', label='Calibrated')
            axes[1, i].plot([0, 1], [0, 1], 'k--', label='Perfect')
            axes[1, i].set_xlabel('Mean Predicted Probability')
            axes[1, i].set_ylabel('Fraction of Positives')
            axes[1, i].set_title(f'Calibrated - {name}')
            axes[1, i].legend()

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    return results


def run_calibration_analysis(
    logits: np.ndarray,
    labels: np.ndarray,
    output_dir: Optional[str] = None,
) -> CalibrationResult:
    """
    Run complete calibration analysis.

    Args:
        logits: Raw logits from sidecar [N, 3]
        labels: True labels [N]
        output_dir: Optional directory for plots

    Returns:
        CalibrationResult with all metrics
    """
    # Convert to tensors for temperature scaling
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)

    # Uncalibrated probabilities
    probs_before = torch.softmax(logits_t, dim=1).numpy()

    # Fit temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(logits_t, labels_t)

    # Calibrated probabilities
    scaled_logits = temp_scaler(logits_t)
    probs_after = torch.softmax(scaled_logits, dim=1).detach().numpy()

    # Compute metrics before calibration
    ece_before = expected_calibration_error(probs_before, labels)
    brier_before = np.mean([
        brier_score_loss((labels == i).astype(int), probs_before[:, i])
        for i in range(3)
    ])
    log_loss_before = log_loss(labels, probs_before, labels=[0, 1, 2])

    # Compute metrics after calibration
    ece_after = expected_calibration_error(probs_after, labels)
    brier_after = np.mean([
        brier_score_loss((labels == i).astype(int), probs_after[:, i])
        for i in range(3)
    ])
    log_loss_after = log_loss(labels, probs_after, labels=[0, 1, 2])

    # ROC analysis
    roc_results = generate_roc_curves(
        probs_after, labels,
        save_path=f"{output_dir}/roc_curves.png" if output_dir else None
    )

    # PR analysis
    pr_results = generate_pr_curves(
        probs_after, labels,
        save_path=f"{output_dir}/pr_curves.png" if output_dir else None
    )

    # Calibration plots
    if output_dir:
        generate_calibration_plot(
            probs_before, labels, probs_after,
            save_path=f"{output_dir}/calibration_plot.png"
        )

    # Cost-weighted evaluation
    predictions = probs_after.argmax(axis=1)
    cost_weighted_error = compute_cost_weighted_error(predictions, labels)

    return CalibrationResult(
        optimal_temperature=optimal_temp,
        ece_before=ece_before,
        ece_after=ece_after,
        brier_before=brier_before,
        brier_after=brier_after,
        log_loss_before=log_loss_before,
        log_loss_after=log_loss_after,
        roc_auc_attack=roc_results["ATTACK"]["auc"],
        roc_auc_safe=roc_results["SAFE"]["auc"],
        pr_auc_attack=pr_results["ATTACK"]["average_precision"],
        pr_auc_safe=pr_results["SAFE"]["average_precision"],
        cost_weighted_error=cost_weighted_error,
    )


def main():
    parser = argparse.ArgumentParser(description="TRYLOCK Sidecar Calibration")
    parser.add_argument("--logits-path", required=True, help="Path to saved logits (npz)")
    parser.add_argument("--labels-path", required=True, help="Path to labels (npz)")
    parser.add_argument("--output-dir", default="calibration_results", help="Output directory")
    args = parser.parse_args()

    # Load data
    logits = np.load(args.logits_path)["logits"]
    labels = np.load(args.labels_path)["labels"]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis
    print("Running calibration analysis...")
    result = run_calibration_analysis(logits, labels, str(output_dir))

    # Print results
    print("\n" + "=" * 60)
    print("SIDECAR CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Optimal Temperature: {result.optimal_temperature:.4f}")
    print()
    print("Before Calibration:")
    print(f"  ECE: {result.ece_before:.4f}")
    print(f"  Brier Score: {result.brier_before:.4f}")
    print(f"  Log Loss: {result.log_loss_before:.4f}")
    print()
    print("After Calibration:")
    print(f"  ECE: {result.ece_after:.4f}")
    print(f"  Brier Score: {result.brier_after:.4f}")
    print(f"  Log Loss: {result.log_loss_after:.4f}")
    print()
    print("ROC AUC:")
    print(f"  ATTACK: {result.roc_auc_attack:.4f}")
    print(f"  SAFE: {result.roc_auc_safe:.4f}")
    print()
    print("Precision-Recall AUC:")
    print(f"  ATTACK: {result.pr_auc_attack:.4f}")
    print(f"  SAFE: {result.pr_auc_safe:.4f}")
    print()
    print(f"Cost-Weighted Error: {result.cost_weighted_error:.4f}")
    print("=" * 60)

    # Save results
    with open(output_dir / "calibration_results.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
