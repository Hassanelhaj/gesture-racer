#!/usr/bin/env python3
"""Train the GestureRacer gesture classifier (12 -> 32 -> 16 -> 3 MLP).

Usage:
    python ml/train.py --data ml/data            # all CSVs in the folder
    python ml/train.py --data "ml/data/*.csv" --epochs 400 --seed 7

Outputs (in --out, default ml/checkpoints):
    gesture_mlp.pt   weights + the normalisation statistics + split settings
    metrics.json     everything printed below, for the README metrics table

Key design choices (also in docs/decisions.md):
  * Splits are by TAKE, never by row, to avoid leakage (see dataset.group_split).
  * Augmentation (noise + small roll jitter) is applied to raw features on the
    fly, because a few recording sessions never cover every hand and camera.
  * Loss is class-weighted so an unbalanced recording session cannot bias it.
  * Accuracy is reported twice: plain, and "gated" at the app's 0.65
    confidence threshold, because that is what a player actually experiences.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dataset import FEATURE_DIM, FEATURE_NAMES, GESTURES, group_split, load_csvs, resolve_csv_paths

TARGET_ACCURACY = 0.92  # from the project brief
APP_MIN_CONFIDENCE = 0.65  # keep equal to CLASSIFIER.minConfidence in src/constants.ts
HIDDEN = (32, 16)

# Augmentation strength (assumptions, tuned by eye on feature ranges; all in raw feature units).
NOISE_STD = 0.02
ROLL_JITTER_RAD = 0.06


class GestureMLP(nn.Module):
    """12 -> 32 -> 16 -> 3. Expects *normalised* inputs; export folds the scaling away."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, HIDDEN[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN[1], len(GESTURES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment(x: torch.Tensor) -> torch.Tensor:
    """Noise + a small coherent rotation of (roll_sin, roll_cos); keeps features in range."""
    out = x.clone()
    out += torch.randn_like(out) * NOISE_STD
    delta = (torch.rand(out.shape[0]) * 2 - 1) * ROLL_JITTER_RAD
    sin, cos = x[:, 0], x[:, 1]
    # sin(a + d) and cos(a + d) from the stored sin(a), cos(a)
    out[:, 0] = sin * torch.cos(delta) + cos * torch.sin(delta) + torch.randn(out.shape[0]) * NOISE_STD
    out[:, 1] = cos * torch.cos(delta) - sin * torch.sin(delta) + torch.randn(out.shape[0]) * NOISE_STD
    out[:, :2] = out[:, :2].clamp(-1, 1)
    out[:, 2:] = out[:, 2:].clamp(0, 1)
    return out


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    """Accuracy, per-class precision/recall, confusion matrix and gated accuracy."""
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    confidence, pred = probs.max(dim=1)
    pred_np, y_np, conf_np = pred.numpy(), y.numpy(), confidence.numpy()

    n_classes = len(GESTURES)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)  # rows = truth, cols = prediction
    for truth, guess in zip(y_np, pred_np):
        confusion[truth, guess] += 1

    per_class = {}
    for i, name in enumerate(GESTURES):
        tp = confusion[i, i]
        precision = tp / max(1, confusion[:, i].sum())
        recall = tp / max(1, confusion[i, :].sum())
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[name] = {"precision": float(precision), "recall": float(recall), "f1": float(f1), "support": int(confusion[i, :].sum())}

    accepted = conf_np >= APP_MIN_CONFIDENCE
    gated_accuracy = float((pred_np[accepted] == y_np[accepted]).mean()) if accepted.any() else 0.0
    return {
        "samples": int(len(y_np)),
        "accuracy": float((pred_np == y_np).mean()),
        "gated_accuracy": gated_accuracy,
        "gated_coverage": float(accepted.mean()),  # share of frames the app would act on
        "per_class": per_class,
        "confusion": confusion.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="ml/data", help="folder, file or glob of CSV recordings")
    parser.add_argument("--out", default="ml/checkpoints", help="output folder")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60, help="stop after this many epochs without a better val loss")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()

    seed_everything(args.seed)
    paths = resolve_csv_paths(args.data)
    x_all, y_all, groups = load_csvs(paths)
    print(f"Loaded {len(y_all)} frames from {len(paths)} file(s): " + ", ".join(f"{g}={int((y_all == i).sum())}" for i, g in enumerate(GESTURES)))

    train_idx, val_idx, test_idx = group_split(y_all, groups, tuple(args.fractions), args.seed)
    print(f"Split by take -> train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)} frames "
          f"({len(np.unique(groups[train_idx]))}/{len(np.unique(groups[val_idx]))}/{len(np.unique(groups[test_idx]))} takes)")

    x_train, y_train = torch.from_numpy(x_all[train_idx]), torch.from_numpy(y_all[train_idx])
    x_val, y_val = torch.from_numpy(x_all[val_idx]), torch.from_numpy(y_all[val_idx])
    x_test, y_test = torch.from_numpy(x_all[test_idx]), torch.from_numpy(y_all[test_idx])

    # Normalisation statistics come from the TRAIN split only.
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-3)
    normalise = lambda t: (t - mean) / std  # noqa: E731

    counts = torch.bincount(y_train, minlength=len(GESTURES)).float()
    class_weights = counts.sum() / (len(GESTURES) * counts.clamp_min(1))
    model = GestureMLP()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=15)

    best_val, best_state, stale = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(y_train))
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(normalise(augment(x_train[batch]))), y_train[batch])
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(normalise(x_val)), y_val).item()
        scheduler.step(val_loss)
        if val_loss < best_val - 1e-5:
            best_val, stale = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch % 25 == 0 or epoch == 1:
            val_acc = (model(normalise(x_val)).argmax(1) == y_val).float().mean().item()
            print(f"epoch {epoch:4d}  val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  lr {optimiser.param_groups[0]['lr']:.5f}")
        if stale >= args.patience:
            print(f"early stop at epoch {epoch} (best val_loss {best_val:.4f})")
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    # Wrap so evaluate() sees raw (un-normalised) features, like the app does.
    class Raw(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            return model(normalise(x))

    results = {name: evaluate(Raw(), x, y) for name, (x, y) in {"train": (x_train, y_train), "val": (x_val, y_val), "test": (x_test, y_test)}.items()}

    print("\nResults")
    print(f"{'split':6} {'frames':>7} {'accuracy':>9} {'gated acc':>10} {'coverage':>9}")
    for name, r in results.items():
        print(f"{name:6} {r['samples']:7d} {r['accuracy']:9.3f} {r['gated_accuracy']:10.3f} {r['gated_coverage']:9.3f}")
    test = results["test"]
    print("\nTest per-class (precision / recall / F1):")
    for name, r in test["per_class"].items():
        print(f"  {name:6} {r['precision']:.3f} / {r['recall']:.3f} / {r['f1']:.3f}  (n={r['support']})")
    print("Test confusion matrix (rows = truth, cols = prediction; order " + ", ".join(GESTURES) + "):")
    for name, row in zip(GESTURES, test["confusion"]):
        print(f"  {name:6} {row}")
    verdict = "PASS" if test["accuracy"] > TARGET_ACCURACY else "BELOW TARGET"
    print(f"\nTest accuracy {test['accuracy']:.3f} vs target > {TARGET_ACCURACY:.2f}: {verdict}")
    if verdict != "PASS":
        print("Tip: more takes in varied distances/positions/lighting help far more than more frames of the same take.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mean": mean,
            "std": std,
            "feature_names": FEATURE_NAMES,
            "gestures": GESTURES,
            "args": vars(args),
        },
        out_dir / "gesture_mlp.pt",
    )
    (out_dir / "metrics.json").write_text(json.dumps({"args": vars(args), "results": results, "target_accuracy": TARGET_ACCURACY}, indent=2))
    print(f"\nSaved {out_dir / 'gesture_mlp.pt'} and metrics.json")


if __name__ == "__main__":
    main()
