#!/usr/bin/env python3
"""
TRYLOCK Layer 2: Representation Engineering Training

Trains linear probes to identify the "attack compliance" direction in
the model's latent space. This direction is then used at runtime to
steer the model away from complying with attacks.

Based on:
- Representation Engineering (Zou et al., 2023): https://arxiv.org/abs/2310.01405
- Circuit Breakers (Zou et al., 2024): https://arxiv.org/abs/2406.04313
- Contrastive Activation Addition (Rimsky et al., 2024)

The key insight: There exists a linear direction in activation space that
separates "willing to comply with attack" from "refusing/deflecting attack".
By identifying this direction, we can intervene during inference to
dampen attack compliance without explicit refusal training.

Usage:
    python training/train_repe.py \
        --activations_dir activations \
        --output_dir outputs/repe \
        --method difference  # or 'probe', 'pca'
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor
from safetensors.torch import save_file, load_file
from tqdm import tqdm

try:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class SteeringVector:
    """
    A steering vector for a specific layer.

    The vector points in the direction of "attack compliance" in activation space.
    Subtracting (scaled by alpha) this vector from activations during inference
    reduces the model's tendency to comply with attacks.
    """
    layer_idx: int
    vector: Tensor  # Shape: [hidden_dim]
    method: str  # How this vector was computed
    train_accuracy: float
    val_accuracy: float
    val_auc: Optional[float] = None

    # Statistics for normalization
    mean_norm: float = 1.0
    std_norm: float = 0.1


@dataclass
class RepEConfig:
    """Configuration for RepE training and inference."""

    # Which layers to create steering vectors for
    target_layers: list[int]

    # Method for computing steering vectors
    method: Literal["difference", "probe", "pca"] = "difference"

    # For probe method: regularization strength
    probe_C: float = 1.0

    # Recommended alpha ranges for different security modes
    alpha_research: float = 0.0   # No steering
    alpha_balanced: float = 1.0   # Default enterprise
    alpha_elevated: float = 1.5   # Heightened security
    alpha_lockdown: float = 2.5   # Maximum protection


def compute_mean_difference_vector(
    attack_activations: Tensor,
    benign_activations: Tensor,
) -> tuple[Tensor, float, float]:
    """
    Compute steering vector using mean difference method.

    This is the simplest and often most effective approach:
    vector = mean(attack_activations) - mean(benign_activations)

    The resulting vector points from "benign behavior" toward "attack compliance".

    Returns:
        - steering_vector: Tensor of shape [hidden_dim]
        - train_acc: Approximate classification accuracy
        - mean_norm: Mean activation norm (for scaling)
    """
    # Compute means
    attack_mean = attack_activations.mean(dim=0)
    benign_mean = benign_activations.mean(dim=0)

    # Steering vector points toward attack compliance
    steering_vector = attack_mean - benign_mean

    # Normalize to unit length
    norm = steering_vector.norm()
    if norm > 0:
        steering_vector = steering_vector / norm

    # Compute approximate accuracy using projection
    all_acts = torch.cat([attack_activations, benign_activations], dim=0)
    all_labels = torch.cat([
        torch.ones(len(attack_activations)),
        torch.zeros(len(benign_activations)),
    ])

    # Project onto steering direction
    projections = (all_acts @ steering_vector).float()
    threshold = projections.median()
    predictions = (projections > threshold).float()
    accuracy = (predictions == all_labels).float().mean().item()

    # Mean norm of activations (for scaling reference)
    mean_norm = all_acts.norm(dim=1).mean().item()

    return steering_vector, accuracy, mean_norm


def compute_probe_vector(
    attack_activations: Tensor,
    benign_activations: Tensor,
    C: float = 1.0,
    test_size: float = 0.2,
) -> tuple[Tensor, float, float, float, float]:
    """
    Compute steering vector using logistic regression probe.

    Train a linear classifier to distinguish attack from benign activations.
    The classifier's weight vector IS the steering direction.

    This method typically yields higher classification accuracy than
    mean difference, but may overfit to spurious features.

    Returns:
        - steering_vector: Tensor of shape [hidden_dim]
        - train_acc: Training accuracy
        - val_acc: Validation accuracy
        - val_auc: Validation AUC-ROC
        - mean_norm: Mean activation norm
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn required for probe method")

    # Prepare data
    X = torch.cat([attack_activations, benign_activations], dim=0).numpy()
    y = np.concatenate([
        np.ones(len(attack_activations)),
        np.zeros(len(benign_activations)),
    ])

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Train logistic regression
    clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Extract weight vector (the steering direction)
    steering_vector = torch.tensor(clf.coef_[0], dtype=torch.float32)

    # Normalize
    norm = steering_vector.norm()
    if norm > 0:
        steering_vector = steering_vector / norm

    # Compute metrics
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    mean_norm = np.linalg.norm(X, axis=1).mean()

    return steering_vector, train_acc, val_acc, val_auc, mean_norm


def compute_pca_vector(
    attack_activations: Tensor,
    benign_activations: Tensor,
    n_components: int = 1,
) -> tuple[Tensor, float, float]:
    """
    Compute steering vector using PCA on the difference.

    Find the principal direction of variance between attack and benign
    activation distributions. This can capture more subtle differences
    than mean difference.

    Returns:
        - steering_vector: Tensor of shape [hidden_dim]
        - explained_variance: Fraction of variance explained
        - mean_norm: Mean activation norm
    """
    # Compute centered activations
    attack_centered = attack_activations - attack_activations.mean(dim=0)
    benign_centered = benign_activations - benign_activations.mean(dim=0)

    # Stack centered activations
    all_centered = torch.cat([attack_centered, benign_centered], dim=0)

    # Compute covariance matrix
    cov = (all_centered.T @ all_centered) / (len(all_centered) - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Top principal component (largest eigenvalue)
    steering_vector = eigenvectors[:, -1]

    # Ensure it points toward attack (positive correlation with attack mean)
    attack_mean = attack_activations.mean(dim=0)
    benign_mean = benign_activations.mean(dim=0)
    diff = attack_mean - benign_mean

    if (steering_vector @ diff) < 0:
        steering_vector = -steering_vector

    # Explained variance
    total_var = eigenvalues.sum()
    explained_var = (eigenvalues[-1] / total_var).item() if total_var > 0 else 0

    # Compute approximate accuracy
    all_acts = torch.cat([attack_activations, benign_activations], dim=0)
    all_labels = torch.cat([
        torch.ones(len(attack_activations)),
        torch.zeros(len(benign_activations)),
    ])

    projections = (all_acts @ steering_vector).float()
    threshold = projections.median()
    predictions = (projections > threshold).float()
    accuracy = (predictions == all_labels).float().mean().item()

    mean_norm = all_acts.norm(dim=1).mean().item()

    return steering_vector, accuracy, mean_norm


def load_activations(activations_dir: Path, layer_idx: int) -> tuple[Tensor, Tensor]:
    """Load attack and benign activations for a specific layer."""
    attack_path = activations_dir / f"attack_layer{layer_idx}.safetensors"
    benign_path = activations_dir / f"benign_layer{layer_idx}.safetensors"

    attack_data = load_file(str(attack_path))
    benign_data = load_file(str(benign_path))

    return attack_data["activations"], benign_data["activations"]


def train_steering_vectors(
    activations_dir: Path,
    target_layers: list[int],
    method: str = "difference",
    probe_C: float = 1.0,
) -> list[SteeringVector]:
    """
    Train steering vectors for all target layers.

    Returns list of SteeringVector objects.
    """
    steering_vectors = []

    for layer_idx in tqdm(target_layers, desc="Training steering vectors"):
        try:
            # Load activations
            attack_acts, benign_acts = load_activations(activations_dir, layer_idx)

            print(f"\n  Layer {layer_idx}:")
            print(f"    Attack samples: {len(attack_acts)}")
            print(f"    Benign samples: {len(benign_acts)}")

            # Compute steering vector based on method
            if method == "difference":
                vector, accuracy, mean_norm = compute_mean_difference_vector(
                    attack_acts, benign_acts
                )
                sv = SteeringVector(
                    layer_idx=layer_idx,
                    vector=vector,
                    method=method,
                    train_accuracy=accuracy,
                    val_accuracy=accuracy,  # Same for difference method
                    mean_norm=mean_norm,
                )

            elif method == "probe":
                vector, train_acc, val_acc, val_auc, mean_norm = compute_probe_vector(
                    attack_acts, benign_acts, C=probe_C
                )
                sv = SteeringVector(
                    layer_idx=layer_idx,
                    vector=vector,
                    method=method,
                    train_accuracy=train_acc,
                    val_accuracy=val_acc,
                    val_auc=val_auc,
                    mean_norm=mean_norm,
                )

            elif method == "pca":
                vector, accuracy, mean_norm = compute_pca_vector(
                    attack_acts, benign_acts
                )
                sv = SteeringVector(
                    layer_idx=layer_idx,
                    vector=vector,
                    method=method,
                    train_accuracy=accuracy,
                    val_accuracy=accuracy,
                    mean_norm=mean_norm,
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            print(f"    Train accuracy: {sv.train_accuracy:.1%}")
            print(f"    Val accuracy: {sv.val_accuracy:.1%}")
            if sv.val_auc:
                print(f"    Val AUC: {sv.val_auc:.3f}")

            steering_vectors.append(sv)

        except Exception as e:
            print(f"\n  Error processing layer {layer_idx}: {e}")
            continue

    return steering_vectors


def save_steering_vectors(
    steering_vectors: list[SteeringVector],
    output_dir: Path,
    config: RepEConfig,
):
    """Save steering vectors and configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save vectors as safetensors
    vectors_dict = {}
    for sv in steering_vectors:
        vectors_dict[f"layer_{sv.layer_idx}"] = sv.vector

    vectors_path = output_dir / "steering_vectors.safetensors"
    save_file(vectors_dict, str(vectors_path))
    print(f"  Saved vectors: {vectors_path}")

    # Save metadata and config
    metadata = {
        "config": asdict(config),
        "vectors": [
            {
                "layer_idx": sv.layer_idx,
                "method": sv.method,
                "train_accuracy": sv.train_accuracy,
                "val_accuracy": sv.val_accuracy,
                "val_auc": sv.val_auc,
                "mean_norm": sv.mean_norm,
                "std_norm": sv.std_norm,
            }
            for sv in steering_vectors
        ],
    }

    meta_path = output_dir / "repe_config.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved config: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="TRYLOCK RepE Training")
    parser.add_argument("--activations_dir", type=str, default="activations",
                        help="Directory containing captured activations")
    parser.add_argument("--output_dir", type=str, default="outputs/repe",
                        help="Output directory for steering vectors")
    parser.add_argument("--method", type=str, default="difference",
                        choices=["difference", "probe", "pca"],
                        help="Method for computing steering vectors")
    parser.add_argument("--probe_C", type=float, default=1.0,
                        help="Regularization strength for probe method")
    parser.add_argument("--target_layers", type=str, default=None,
                        help="Override target layers (comma-separated)")

    args = parser.parse_args()

    print("=" * 60)
    print("TRYLOCK REPRESENTATION ENGINEERING TRAINING")
    print("=" * 60)
    print(f"Activations: {args.activations_dir}")
    print(f"Method: {args.method}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    activations_dir = Path(args.activations_dir)
    output_dir = Path(args.output_dir)

    # Load metadata to get target layers
    meta_path = activations_dir / "metadata.json"
    with open(meta_path) as f:
        act_metadata = json.load(f)

    target_layers = act_metadata["target_layers"]
    if args.target_layers:
        target_layers = [int(x.strip()) for x in args.target_layers.split(",")]

    print(f"\nTarget layers: {target_layers}")
    print(f"Total samples in activations: {act_metadata['total_samples']}")
    print(f"Attack samples: {act_metadata['attack_count']}")
    print(f"Benign samples: {act_metadata['benign_count']}")

    # Train steering vectors
    print("\n" + "-" * 60)
    print("TRAINING STEERING VECTORS")
    print("-" * 60)

    steering_vectors = train_steering_vectors(
        activations_dir,
        target_layers,
        method=args.method,
        probe_C=args.probe_C,
    )

    # Create config
    config = RepEConfig(
        target_layers=target_layers,
        method=args.method,
        probe_C=args.probe_C,
    )

    # Save
    print("\n" + "-" * 60)
    print("SAVING OUTPUTS")
    print("-" * 60)

    save_steering_vectors(steering_vectors, output_dir, config)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("\nSteering Vector Summary:")
    for sv in steering_vectors:
        auc_str = f", AUC: {sv.val_auc:.3f}" if sv.val_auc else ""
        print(f"  Layer {sv.layer_idx}: "
              f"val_acc={sv.val_accuracy:.1%}{auc_str}")

    avg_acc = sum(sv.val_accuracy for sv in steering_vectors) / len(steering_vectors)
    print(f"\nAverage validation accuracy: {avg_acc:.1%}")

    print("\nRecommended alpha values:")
    print(f"  Research mode (no steering):     {config.alpha_research}")
    print(f"  Balanced mode (default):         {config.alpha_balanced}")
    print(f"  Elevated mode (heightened):      {config.alpha_elevated}")
    print(f"  Lockdown mode (max protection):  {config.alpha_lockdown}")

    print(f"\nOutputs saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  1. Test steering: python training/repe_steering.py --vectors {output_dir} --alpha 1.0")
    print(f"  2. Integrate with model inference pipeline")


if __name__ == "__main__":
    main()
