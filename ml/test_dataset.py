"""Tests for dataset.py (numpy only, no torch needed). Run: python -m unittest ml/test_dataset.py -v"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dataset import FEATURE_NAMES, GESTURES, group_split, load_csvs, resolve_csv_paths  # noqa: E402

HEADER = ",".join(["gesture", "group", *FEATURE_NAMES])


def write_csv(path: Path, takes: dict[str, str], frames: int = 5, nan_row: bool = False) -> None:
    """takes: {group_name: gesture}."""
    lines = [HEADER]
    for group, gesture in takes.items():
        for i in range(frames):
            values = ",".join(f"{(i + 1) / 10:.3f}" for _ in FEATURE_NAMES)
            lines.append(f"{gesture},{group},{values}")
    if nan_row:
        lines.append("LEFT,bad," + ",".join(["nan"] * len(FEATURE_NAMES)))
    path.write_text("\n".join(lines) + "\n")


def full_takes(per_class: int) -> dict[str, str]:
    return {f"{g[0]}{i}": g for g in GESTURES for i in range(per_class)}


class LoadTests(unittest.TestCase):
    def test_loads_and_prefixes_groups_with_file_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            write_csv(a, {"t1": "LEFT"})
            write_csv(b, {"t1": "RIGHT"})
            x, y, groups = load_csvs(resolve_csv_paths(tmp))
            self.assertEqual(x.shape, (10, len(FEATURE_NAMES)))
            self.assertEqual(sorted(set(groups)), ["a:t1", "b:t1"])  # same take name, different files
            self.assertEqual(sorted(set(y.tolist())), [GESTURES.index("LEFT"), GESTURES.index("RIGHT")])

    def test_rejects_a_header_that_disagrees_with_the_app(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.csv"
            path.write_text("gesture,group,roll_sin\nLEFT,t1,0.1\n")
            with self.assertRaisesRegex(ValueError, "header does not match"):
                load_csvs([path])

    def test_rejects_unknown_gesture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            write_csv(path, {"t1": "LEFT"})
            path.write_text(path.read_text().replace("LEFT", "WAVE", 1))
            with self.assertRaisesRegex(ValueError, "unknown gesture"):
                load_csvs([path])

    def test_drops_non_finite_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            write_csv(path, {"t1": "LEFT"}, nan_row=True)
            x, _, _ = load_csvs([path])
            self.assertEqual(len(x), 5)

    def test_missing_data_has_a_helpful_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "npm run collect"):
                resolve_csv_paths(tmp)


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "d.csv"
        write_csv(path, full_takes(8), frames=10)
        self.x, self.y, self.groups = load_csvs([path])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_no_take_is_split_across_partitions(self) -> None:
        train, val, test = group_split(self.y, self.groups, (0.7, 0.15, 0.15), seed=1)
        sets = [set(self.groups[idx]) for idx in (train, val, test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_partitions_cover_everything_once(self) -> None:
        parts = group_split(self.y, self.groups, (0.7, 0.15, 0.15), seed=1)
        merged = np.sort(np.concatenate(parts))
        np.testing.assert_array_equal(merged, np.arange(len(self.y)))

    def test_every_split_sees_every_gesture(self) -> None:
        for idx in group_split(self.y, self.groups, (0.7, 0.15, 0.15), seed=3):
            self.assertEqual(set(self.y[idx].tolist()), {0, 1, 2})

    def test_is_deterministic_per_seed_and_differs_across_seeds(self) -> None:
        a = group_split(self.y, self.groups, (0.7, 0.15, 0.15), seed=5)
        b = group_split(self.y, self.groups, (0.7, 0.15, 0.15), seed=5)
        c = group_split(self.y, self.groups, (0.7, 0.15, 0.15), seed=6)
        for left, right in zip(a, b):
            np.testing.assert_array_equal(left, right)
        self.assertFalse(all(np.array_equal(l, r) for l, r in zip(a, c)))

    def test_too_few_takes_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "few.csv"
            write_csv(path, full_takes(2))
            _, y, groups = load_csvs([path])
            with self.assertRaisesRegex(ValueError, "need at least 3"):
                group_split(y, groups, (0.7, 0.15, 0.15), seed=0)

    def test_fractions_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            group_split(self.y, self.groups, (0.5, 0.2, 0.2), seed=0)


if __name__ == "__main__":
    unittest.main()
