#!/usr/bin/env python3
"""Export the trained classifier to ONNX and quantise it to INT8.

Usage:
    python ml/export_onnx.py                      # writes public/models/gesture.int8.onnx
    python ml/export_onnx.py --data ml/data       # also reports INT8 accuracy on the held-out test takes

What it does, and why:
  1. Folds the input normalisation into the first layer, so the browser feeds raw
     features to a graph that is just MatMul/Add/Relu. No scaler to keep in sync.
  2. Exports with MatMul + Add (not Gemm), because ONNX Runtime's dynamic
     quantiser only converts MatMul weights.
  3. Dynamic INT8 quantisation, as the project brief requires.
  4. Verifies fp32-ONNX == PyTorch, then INT8 vs fp32 agreement, and measures size
     and CPU latency, so the README's metrics are measured, not assumed.

Model contract with the app (src/vision/classifier.ts):
    input  "features"  float32 [batch, 12]   raw features, in FEATURE_NAMES order
    output "logits"    float32 [batch, 3]    LEFT, RIGHT, JUMP (softmax is done in TypeScript)
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from dataset import FEATURE_DIM, FEATURE_NAMES, GESTURES, group_split, load_csvs, resolve_csv_paths

INPUT_NAME = "features"
OUTPUT_NAME = "logits"
OPSET = 17
FP32_TOLERANCE = 1e-4  # max |torch - onnx fp32| allowed on logits
MIN_INT8_AGREEMENT = 0.99  # warn below this top-1 agreement with fp32
LATENCY_RUNS = 2000


class FoldedMLP(nn.Module):
    """Plain 12-32-16-3 MLP written with `x @ W + b` so it exports as MatMul + Add."""

    def __init__(self, linears: list[nn.Linear], mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        first, *rest = linears
        # (x - mean) / std @ W^T + b  ==  x @ (W^T / std[:, None]) + (b - (mean / std) @ W^T)
        w1 = first.weight.detach().T / std[:, None]
        b1 = first.bias.detach() - (mean / std) @ first.weight.detach().T
        self.weights = nn.ParameterList([nn.Parameter(w1.clone())] + [nn.Parameter(l.weight.detach().T.clone()) for l in rest])
        self.biases = nn.ParameterList([nn.Parameter(b1.clone())] + [nn.Parameter(l.bias.detach().clone()) for l in rest])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < last:
                x = torch.relu(x)
        return x


def load_checkpoint(path: Path) -> tuple[list[nn.Linear], torch.Tensor, torch.Tensor, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["feature_names"] != FEATURE_NAMES or ckpt["gestures"] != GESTURES:
        raise ValueError("Checkpoint was trained with different features/gestures than dataset.py declares.")
    state = ckpt["state_dict"]
    linears = []
    for index in (0, 2, 4):  # Sequential positions of the three Linear layers
        weight, bias = state[f"net.{index}.weight"], state[f"net.{index}.bias"]
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        layer.weight.data, layer.bias.data = weight, bias
        linears.append(layer)
    return linears, ckpt["mean"], ckpt["std"], ckpt["args"]


def reference_logits(linears: list[nn.Linear], mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The un-folded forward pass exactly as it was trained."""
    h = (x - mean) / std
    for i, layer in enumerate(linears):
        h = layer(h)
        if i < len(linears) - 1:
            h = torch.relu(h)
    return h


def export_fp32(model: nn.Module, path: Path) -> None:
    dummy = torch.zeros(1, FEATURE_DIM)
    kwargs = dict(
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=OPSET,
    )
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)  # torch >= 2.5: opt out of the dynamo exporter
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kwargs)  # older torch has no `dynamo` argument
    onnx.checker.check_model(onnx.load(str(path)))


def run(path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run([OUTPUT_NAME], {INPUT_NAME: x.astype(np.float32)})[0]


def latency_ms(path: Path) -> dict:
    """Single-sample CPU latency through onnxruntime (python). Browser numbers differ; use the in-app Diagnostics panel for those."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    sample = np.random.default_rng(0).random((1, FEATURE_DIM), dtype=np.float32)
    for _ in range(100):
        session.run([OUTPUT_NAME], {INPUT_NAME: sample})
    times = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        session.run([OUTPUT_NAME], {INPUT_NAME: sample})
        times.append((time.perf_counter() - start) * 1000)
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="ml/checkpoints/gesture_mlp.pt")
    parser.add_argument("--out-dir", default="public/models", help="where gesture.int8.onnx is written (the app loads it from here)")
    parser.add_argument("--work-dir", default="ml/checkpoints", help="where the fp32 model and report go")
    parser.add_argument("--data", default=None, help="CSV folder/glob; enables the INT8 test-split accuracy check")
    parser.add_argument("--also-fp32", action="store_true", help="also write gesture.fp32.onnx next to the INT8 model (see docs/decisions.md on INT8 vs fp32)")
    args = parser.parse_args()

    linears, mean, std, train_args = load_checkpoint(Path(args.checkpoint))
    folded = FoldedMLP(linears, mean, std).eval()

    # Folding must be exact: compare against the model as trained, on random in-range inputs.
    rng = np.random.default_rng(0)
    probe = torch.from_numpy(rng.random((512, FEATURE_DIM), dtype=np.float32))
    with torch.no_grad():
        fold_error = (folded(probe) - reference_logits(linears, mean, std, probe)).abs().max().item()
    if fold_error > 1e-4:
        raise SystemExit(f"Folding the normalisation changed the outputs (max diff {fold_error:.2e}).")

    work_dir, out_dir = Path(args.work_dir), Path(args.out_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path, int8_path = work_dir / "gesture.fp32.onnx", out_dir / "gesture.int8.onnx"

    export_fp32(folded, fp32_path)
    onnx_error = float(np.abs(run(fp32_path, probe.numpy()) - folded(probe).detach().numpy()).max())
    if onnx_error > FP32_TOLERANCE:
        raise SystemExit(f"ONNX fp32 output differs from PyTorch (max diff {onnx_error:.2e}).")

    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    onnx.checker.check_model(onnx.load(str(int8_path)))
    if args.also_fp32:
        shutil.copy(fp32_path, out_dir / "gesture.fp32.onnx")
    op_types = sorted({node.op_type for node in onnx.load(str(int8_path)).graph.node})

    fp32_pred = run(fp32_path, probe.numpy()).argmax(1)
    int8_pred = run(int8_path, probe.numpy()).argmax(1)
    agreement = float((fp32_pred == int8_pred).mean())

    report: dict = {
        "contract": {"input": INPUT_NAME, "output": OUTPUT_NAME, "features": FEATURE_NAMES, "gestures": GESTURES},
        "fold_max_error": fold_error,
        "onnx_fp32_max_error": onnx_error,
        "int8_top1_agreement_random_inputs": agreement,
        "int8_ops": op_types,
        "size_bytes": {"fp32": fp32_path.stat().st_size, "int8": int8_path.stat().st_size},
        "latency_ms_python_cpu": {"fp32": latency_ms(fp32_path), "int8": latency_ms(int8_path)},
    }

    if args.data:
        x_all, y_all, groups = load_csvs(resolve_csv_paths(args.data))
        _, _, test_idx = group_split(y_all, groups, tuple(train_args["fractions"]), train_args["seed"])
        x_test, y_test = x_all[test_idx], y_all[test_idx]
        fp32_acc = float((run(fp32_path, x_test).argmax(1) == y_test).mean())
        int8_acc = float((run(int8_path, x_test).argmax(1) == y_test).mean())
        report["test_accuracy"] = {"fp32": fp32_acc, "int8": int8_acc, "frames": int(len(y_test))}

    (work_dir / "export_report.json").write_text(json.dumps(report, indent=2))

    print(f"Wrote {int8_path} ({report['size_bytes']['int8']:,} bytes; fp32 is {report['size_bytes']['fp32']:,})")
    print(f"  normalisation fold error {fold_error:.1e}, ONNX fp32 vs PyTorch {onnx_error:.1e}")
    print(f"  INT8 ops: {', '.join(op_types)}")
    print(f"  INT8 vs fp32 top-1 agreement on random inputs: {agreement:.3f}")
    if agreement < MIN_INT8_AGREEMENT:
        print("  WARNING: INT8 disagrees with fp32 noticeably; consider shipping the fp32 model instead.")
    print("  Note: INT8 results can differ slightly between ONNX Runtime builds (Python vs browser WASM);")
    print("        fp32 matches across runtimes. Use --also-fp32 and compare in the app's Diagnostics if in doubt.")
    lat = report["latency_ms_python_cpu"]
    print(f"  CPU latency p50/p95 (python): fp32 {lat['fp32']['p50']:.3f}/{lat['fp32']['p95']:.3f} ms, int8 {lat['int8']['p50']:.3f}/{lat['int8']['p95']:.3f} ms")
    if "test_accuracy" in report:
        acc = report["test_accuracy"]
        print(f"  Held-out test accuracy: fp32 {acc['fp32']:.3f}, int8 {acc['int8']:.3f} ({acc['frames']} frames)")


if __name__ == "__main__":
    main()
