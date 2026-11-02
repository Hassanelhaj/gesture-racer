"""Shared dataset code for train.py and export_onnx.py.

The two constants below are a CONTRACT with the TypeScript app:
  GESTURES       <-> `GESTURES` in src/types.ts (index = model output class)
  FEATURE_NAMES  <-> `FEATURE_NAMES` in src/vision/features.ts (order = model input)
The collector writes the feature names into every CSV header and `load_csvs`
refuses a file whose header disagrees, so drift is caught loudly here instead
of silently producing a model that is wrong in the browser.
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np

GESTURES = ["LEFT", "RIGHT", "JUMP"]
FEATURE_NAMES = [
    "roll_sin",
    "roll_cos",
    "wrist_x",
    "wrist_y",
    "palm_scale",
    "curl_thumb",
    "curl_index",
    "curl_middle",
    "curl_ring",
    "curl_pinky",
    "finger_spread",
    "hand_openness",
]
FEATURE_DIM = len(FEATURE_NAMES)
MIN_GROUPS_PER_CLASS = 3  # one take per split, otherwise there is nothing to validate on


def resolve_csv_paths(spec: str) -> list[Path]:
    """Accepts a directory, a single file or a glob and returns the CSV files found."""
    path = Path(spec)
    if path.is_dir():
        found = sorted(path.glob("*.csv"))
    else:
        found = sorted(Path(p) for p in glob.glob(spec))
    if not found:
        raise FileNotFoundError(
            f"No CSV files found for '{spec}'. Record data with `npm run collect` "
            "and save the downloaded files into ml/data/."
        )
    return found


def load_csvs(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads and validates recordings.

    Returns (X float32 [n, 12], y int64 [n], groups str [n]). A *group* is one
    recorded take; frames inside a take are near-duplicates, so splits must
    never separate a take.
    """
    rows_x: list[list[float]] = []
    rows_y: list[int] = []
    rows_g: list[str] = []
    expected = ["gesture", "group", *FEATURE_NAMES]
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != expected:
                raise ValueError(
                    f"{path.name}: header does not match the app's features.\n"
                    f"  expected: {expected}\n  found:    {header}\n"
                    "Re-record with the current collector, or update FEATURE_NAMES in dataset.py."
                )
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(expected):
                    raise ValueError(f"{path.name}:{line_number}: expected {len(expected)} columns")
                gesture, group, *values = row
                if gesture not in GESTURES:
                    raise ValueError(f"{path.name}:{line_number}: unknown gesture '{gesture}'")
                features = [float(v) for v in values]
                if not all(np.isfinite(features)):
                    continue  # a NaN/inf frame is useless; drop it rather than poison training
                rows_x.append(features)
                rows_y.append(GESTURES.index(gesture))
                rows_g.append(f"{path.stem}:{group}")  # prefix the file so groups cannot collide
    if not rows_x:
        raise ValueError("The CSV files contain no usable rows.")
    return (
        np.asarray(rows_x, dtype=np.float32),
        np.asarray(rows_y, dtype=np.int64),
        np.asarray(rows_g),
    )


def group_split(
    y: np.ndarray,
    groups: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits row indices into (train, val, test) at the *take* level.

    WHY: splitting rows randomly would put near-identical neighbouring frames
    of one take into both train and test, inflating accuracy (leakage). Each
    class is split separately so every split sees every gesture.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    rng = np.random.default_rng(seed)
    parts: tuple[list[int], list[int], list[int]] = ([], [], [])
    for cls, name in enumerate(GESTURES):
        class_groups = np.unique(groups[y == cls])
        if len(class_groups) < MIN_GROUPS_PER_CLASS:
            raise ValueError(
                f"{name} has only {len(class_groups)} take(s); need at least {MIN_GROUPS_PER_CLASS}. "
                "Record more takes (each take = one press of Record)."
            )
        rng.shuffle(class_groups)
        n = len(class_groups)
        n_val = max(1, round(n * fractions[1]))
        n_test = max(1, round(n * fractions[2]))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError(f"{name}: not enough takes ({n}) for a train/val/test split")
        chosen = (class_groups[:n_train], class_groups[n_train : n_train + n_val], class_groups[n_train + n_val :])
        for part, selected in zip(parts, chosen):
            part.extend(np.flatnonzero((y == cls) & np.isin(groups, selected)).tolist())
    return tuple(np.asarray(sorted(p), dtype=np.int64) for p in parts)  # type: ignore[return-value]
