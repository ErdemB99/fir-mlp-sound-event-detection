
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

from PFG_parameters import load_config, set_seed
from PFG_neural_network import FIR_MLP


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RESULTS = ROOT / "results"
CHECKPOINTS = RESULTS / "checkpoints"

TRAIN_FEATURES = DATA / "training_par"
TRAIN_LABELS = DATA / "training_target"

EVAL_FEATURES = DATA / "eval_par"
EVAL_LABELS = DATA / "eval_target"

PARAM_FILE = ROOT / "inputParameters.txt"

CHECKPOINTS.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

class AudioDataset(Dataset):

    def __init__(
        self,
        files,
        feature_dir,
        label_dir,
        config,
        statistics=None,
        max_frames=None
    ):
        self.files = list(files)
        self.feature_dir = Path(feature_dir)
        self.label_dir = Path(label_dir)
        self.config = config
        self.statistics = statistics
        self.max_frames = max_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]

        X = np.load(
            self.feature_dir / filename,
            allow_pickle=False
        ).astype(np.float32)

        Y = np.load(
            self.label_dir / filename,
            allow_pickle=False
        ).astype(np.float32)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(
                f"Expected 2D arrays: {filename}"
            )

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Frame mismatch: {filename}"
            )

        if X.shape[0] == 0:
            raise ValueError(
                f"Empty audio clip: {filename}"
            )

        if X.shape[1] != self.config["neurons"][0]:
            raise ValueError(
                f"Invalid feature dimension: {filename}"
            )

        if Y.shape[1] != self.config["neurons"][-1]:
            raise ValueError(
                f"Invalid label dimension: {filename}"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite features: {filename}"
            )

        if not np.isfinite(Y).all():
            raise ValueError(
                f"Non-finite labels: {filename}"
            )

        if not np.all((Y == 0) | (Y == 1)):
            raise ValueError(
                f"Labels must be binary: {filename}"
            )

        if self.max_frames is not None:
            X = X[:self.max_frames]
            Y = Y[:self.max_frames]

        # Normalize using TRAINING statistics only.
        if self.statistics is not None:
            X = (
                X - self.statistics["mean"]
            ) / self.statistics["std"]

        return (
            torch.from_numpy(
                np.ascontiguousarray(X)
            ),
            torch.from_numpy(
                np.ascontiguousarray(Y)
            )
        )


# ============================================================
# FILE VALIDATION
# ============================================================

def get_files(feature_dir, label_dir):

    feature_dir = Path(feature_dir)
    label_dir = Path(label_dir)

    if not feature_dir.is_dir():
        raise FileNotFoundError(feature_dir)

    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)

    feature_files = {
        p.name for p in feature_dir.glob("*.npy")
    }

    label_files = {
        p.name for p in label_dir.glob("*.npy")
    }

    missing_labels = feature_files - label_files
    missing_features = label_files - feature_files

    if missing_labels or missing_features:
        raise ValueError(
            "Feature/label mismatch:\n"
            f"Missing labels: {sorted(missing_labels)[:10]}\n"
            f"Missing features: {sorted(missing_features)[:10]}"
        )

    files = sorted(feature_files & label_files)

    if not files:
        raise ValueError(
            "No matching feature/label files found."
        )

    return files


# ============================================================
# TRAINING STATISTICS
# ============================================================

def compute_training_statistics(
    files,
    feature_dir,
    label_dir,
    config,
    max_frames=None
):
    """
    Calculate per-feature normalization statistics
    and positive class weights.

    IMPORTANT:
    Only training files are used.
    Validation and evaluation data are excluded.
    """

    feature_sum = None
    feature_sq_sum = None
    class_counts = None

    total_frames = 0

    for filename in files:

        X = np.load(
            feature_dir / filename,
            allow_pickle=False
        ).astype(np.float64)

        Y = np.load(
            label_dir / filename,
            allow_pickle=False
        ).astype(np.float64)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(
                f"Invalid array dimensions: {filename}"
            )

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Frame mismatch: {filename}"
            )

        if X.shape[1] != config["neurons"][0]:
            raise ValueError(
                f"Feature dimension mismatch: {filename}"
            )

        if Y.shape[1] != config["neurons"][-1]:
            raise ValueError(
                f"Label dimension mismatch: {filename}"
            )

        if max_frames is not None:
            X = X[:max_frames]
            Y = Y[:max_frames]

        if len(X) == 0:
            raise ValueError(
                f"Empty training clip: {filename}"
            )

        if not np.isfinite(X).all():
            raise ValueError(
                f"Non-finite features: {filename}"
            )

        if not np.isfinite(Y).all():
            raise ValueError(
                f"Non-finite labels: {filename}"
            )

        if not np.all((Y == 0) | (Y == 1)):
            raise ValueError(
                f"Non-binary labels: {filename}"
            )

        if feature_sum is None:
            feature_sum = np.zeros(
                X.shape[1], dtype=np.float64
            )

            feature_sq_sum = np.zeros(
                X.shape[1], dtype=np.float64
            )

            class_counts = np.zeros(
                Y.shape[1], dtype=np.float64
            )

        feature_sum += X.sum(axis=0)

        feature_sq_sum += np.square(X).sum(axis=0)

        class_counts += Y.sum(axis=0)

        total_frames += X.shape[0]

    if total_frames == 0:
        raise ValueError("No training frames available.")

    mean = feature_sum / total_frames

    variance = (
        feature_sq_sum / total_frames
        - np.square(mean)
    )

    std = np.sqrt(
        np.maximum(variance, 1e-8)
    )

    negative_counts = total_frames - class_counts

    # Balanced positive class weighting.
    # Cap extreme values for training stability.
    pos_weight = np.clip(
        negative_counts / np.maximum(class_counts, 1.0),
        1.0,
        20.0
    )

    # Classes absent from training cannot be learned.
    pos_weight[class_counts == 0] = 1.0

    statistics = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "pos_weight": pos_weight.astype(np.float32),
        "class_counts": class_counts.astype(np.float32),
        "total_frames": int(total_frames)
    }

    return statistics


# ============================================================
# MODEL
# ============================================================

def create_model(config, device):

    model = FIR_MLP(
        num_layers=config["num_layers"],
        num_neurons=config["neurons"],
        FIR_lengths=config["fir_lengths"],
        learning_rate=config["learning_rate"],
        momentum=config["momentum"]
    )

    return model.to(device)


# ============================================================
# DATA LOADER
# ============================================================

def create_loader(
    files,
    feature_dir,
    label_dir,
    config,
    statistics,
    max_frames=None,
    shuffle=False
):

    dataset = AudioDataset(
        files=files,
        feature_dir=feature_dir,
        label_dir=label_dir,
        config=config,
        statistics=statistics,
        max_frames=max_frames
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
    threshold=0.5
):

    y_true = np.asarray(
        y_true, dtype=np.int32
    )

    y_pred = (
        np.asarray(y_prob) >= threshold
    ).astype(np.int32)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="micro",
            zero_division=0
        )
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def find_best_threshold(
    y_true,
    y_prob
):
    """
    Select a global probability threshold
    using validation predictions only.
    """

    thresholds = np.arange(
        0.10, 0.91, 0.05
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            y_prob,
            threshold=float(threshold)
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)

    return best_threshold, best_f1


# ============================================================
# TRAINING / EVALUATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    threshold=0.5,
    return_predictions=False
):

    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_elements = 0

    all_labels = []
    all_probabilities = []

    for X, Y in loader:

        X = X.to(device)
        Y = Y.to(device)

        with torch.set_grad_enabled(training):

            logits = model(X)

            loss = criterion(logits, Y)

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

        total_loss += (
            loss.item() * Y.numel()
        )

        total_elements += Y.numel()

        labels = (
            Y.detach()
            .cpu()
            .numpy()
            .reshape(-1, Y.shape[-1])
        )

        probabilities = (
            torch.sigmoid(logits.detach())
            .cpu()
            .numpy()
            .reshape(-1, Y.shape[-1])
        )

        all_labels.append(labels)
        all_probabilities.append(probabilities)

    if not all_labels:
        raise ValueError("Empty data loader.")

    y_true = np.concatenate(
        all_labels, axis=0
    )

    y_prob = np.concatenate(
        all_probabilities, axis=0
    )

    metrics = calculate_metrics(
        y_true,
        y_prob,
        threshold=threshold
    )

    metrics["loss"] = (
        total_loss / max(total_elements, 1)
    )

    if return_predictions:
        return metrics, y_true, y_prob

    return metrics


# ============================================================
# CHECKPOINTS
# ============================================================

def save_checkpoint(
    model,
    config,
    statistics,
    filename,
    epoch,
    score,
    threshold=0.5
):

    path = CHECKPOINTS / filename

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": int(epoch),
            "f1": float(score),
            "threshold": float(threshold),
            "mean": statistics["mean"],
            "std": statistics["std"],
            "pos_weight": statistics["pos_weight"]
        },
        path
    )

    return path


# ============================================================
# TRAINING
# ============================================================

def train_model(
    train_files,
    val_files,
    config,
    args,
    checkpoint_name
):

    statistics = compute_training_statistics(
        files=train_files,
        feature_dir=TRAIN_FEATURES,
        label_dir=TRAIN_LABELS,
        config=config,
        max_frames=args.max_frames
    )

    print("\nTRAINING STATISTICS")
    print("Frames:", statistics["total_frames"])

    print(
        "Positive class weights:",
        np.round(statistics["pos_weight"], 2)
    )

    print(
        "Feature mean range:",
        float(statistics["mean"].min()),
        float(statistics["mean"].max())
    )

    print(
        "Feature std range:",
        float(statistics["std"].min()),
        float(statistics["std"].max())
    )

    model = create_model(
        config,
        args.device
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"]
    )

    pos_weight = torch.tensor(
        statistics["pos_weight"],
        dtype=torch.float32,
        device=args.device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    train_loader = create_loader(
        files=train_files,
        feature_dir=TRAIN_FEATURES,
        label_dir=TRAIN_LABELS,
        config=config,
        statistics=statistics,
        max_frames=args.max_frames,
        shuffle=True
    )

    val_loader = None

    if val_files:

        val_loader = create_loader(
            files=val_files,
            feature_dir=TRAIN_FEATURES,
            label_dir=TRAIN_LABELS,
            config=config,
            statistics=statistics,
            max_frames=args.max_frames,
            shuffle=False
        )

    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    for epoch in range(1, args.epochs + 1):

        start = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=args.device,
            optimizer=optimizer
        )

        if val_loader is not None:

            val_metrics, y_true, y_prob = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=args.device,
                return_predictions=True
            )

            best_threshold, tuned_f1 = find_best_threshold(
                y_true,
                y_prob
            )

            tuned_metrics = calculate_metrics(
                y_true,
                y_prob,
                threshold=best_threshold
            )

            val_metrics["f1"] = tuned_metrics["f1"]
            val_metrics["precision"] = tuned_metrics["precision"]
            val_metrics["recall"] = tuned_metrics["recall"]

        else:

            val_metrics = None
            best_threshold = 0.5

        elapsed = time.time() - start

        if val_metrics is not None:

            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train F1@0.5: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Threshold: {best_threshold:.2f} | "
                f"Time: {elapsed:.1f}s"
            )

        else:

            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train F1@0.5: {train_metrics['f1']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "threshold": best_threshold
        })

        # Final training without validation:
        # Always save the latest model.
        if val_loader is None:

            save_checkpoint(
                model=model,
                config=config,
                statistics=statistics,
                filename=checkpoint_name,
                epoch=epoch,
                score=train_metrics["f1"],
                threshold=0.5
            )

            best_epoch = epoch
            best_f1 = train_metrics["f1"]

            continue

        # Validation-based model selection.
        improved = (
            val_metrics["f1"] > best_f1 + 1e-6
            or (
                abs(val_metrics["f1"] - best_f1) <= 1e-6
                and val_metrics["loss"] < best_loss
            )
        )

        if improved:

            best_f1 = val_metrics["f1"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            save_checkpoint(
                model=model,
                config=config,
                statistics=statistics,
                filename=checkpoint_name,
                epoch=epoch,
                score=best_f1,
                threshold=best_threshold
            )

        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    history_path = RESULTS / (
        Path(checkpoint_name).stem + "_history.json"
    )

    with history_path.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            history,
            f,
            indent=2
        )

    print(
        f"Saved checkpoint: {checkpoint_name} | "
        f"Epoch: {best_epoch} | "
        f"F1: {best_f1:.4f}"
    )

    return best_f1


# ============================================================
# CROSS-VALIDATION
# ============================================================

def cross_validation(config, args):

    files = get_files(
        TRAIN_FEATURES,
        TRAIN_LABELS
    )

    if args.max_files is not None:
        files = files[:args.max_files]

    if len(files) < 5:
        raise ValueError(
            "At least 5 clips required for 5-fold CV."
        )

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=args.seed
    )

    scores = []

    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(files),
        start=1
    ):

        print(f"\n========== FOLD {fold}/5 ==========")

        set_seed(args.seed + fold)

        train_files = [
            files[i] for i in train_idx
        ]

        val_files = [
            files[i] for i in val_idx
        ]

        score = train_model(
            train_files=train_files,
            val_files=val_files,
            config=config,
            args=args,
            checkpoint_name=f"fold_{fold}_best.pt"
        )

        scores.append(score)

    results = {
        "fold_f1": [float(x) for x in scores],
        "mean_f1": float(np.mean(scores)),
        "std_f1": float(np.std(scores))
    }

    with (
        RESULTS / "cross_validation.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print("\nCROSS-VALIDATION RESULTS")

    print(
        f"Mean Micro-F1: {results['mean_f1']:.4f}"
    )

    print(
        f"Std Micro-F1: {results['std_f1']:.4f}"
    )


# ============================================================
# FINAL TRAINING
# ============================================================

def final_training(config, args):

    files = get_files(
        TRAIN_FEATURES,
        TRAIN_LABELS
    )

    if args.max_files is not None:
        files = files[:args.max_files]

    print(
        f"Final training on {len(files)} clips."
    )

    train_model(
        train_files=files,
        val_files=[],
        config=config,
        args=args,
        checkpoint_name="final_model.pt"
    )


# ============================================================
# PREDICTION / EVALUATION
# ============================================================

def prediction(config, args):

    checkpoint_path = CHECKPOINTS / args.checkpoint

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Load only checkpoints from trusted sources.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=args.device,
        weights_only=True
    )

    saved_config = checkpoint["config"]

    statistics = {
        "mean": np.asarray(
            checkpoint["mean"],
            dtype=np.float32
        ),
        "std": np.asarray(
            checkpoint["std"],
            dtype=np.float32
        )
    }

    threshold = float(
        checkpoint.get("threshold", 0.5)
    )

    model = create_model(
        saved_config,
        args.device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    files = get_files(
        EVAL_FEATURES,
        EVAL_LABELS
    )

    loader = create_loader(
        files=files,
        feature_dir=EVAL_FEATURES,
        label_dir=EVAL_LABELS,
        config=saved_config,
        statistics=statistics,
        max_frames=args.max_frames
    )

    criterion = nn.BCEWithLogitsLoss()

    metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=args.device,
        threshold=threshold
    )

    print("\nEVALUATION RESULTS")
    print(f"Threshold: {threshold:.2f}")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    with (
        RESULTS / "evaluation.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "threshold": threshold,
                "metrics": metrics
            },
            f,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="FIR-MLP Sound Event Detection"
    )

    parser.add_argument(
        "mode",
        choices=["1", "2", "3"],
        help="1=Cross-validation, 2=Final training, 3=Evaluation"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final_model.pt"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu"
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    if args.patience < 1:
        parser.error("--patience must be >= 1")

    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be >= 1")

    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be >= 1")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available.")

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available.")

    set_seed(args.seed)

    config = load_config(PARAM_FILE)

    print("\nFIR-MLP SOUND EVENT DETECTION")
    print("Architecture:", config["neurons"])
    print("FIR lengths:", config["fir_lengths"])
    print("Device:", args.device)

    if args.mode == "1":
        cross_validation(config, args)

    elif args.mode == "2":
        final_training(config, args)

    elif args.mode == "3":
        prediction(config, args)


if __name__ == "__main__":
    main()
