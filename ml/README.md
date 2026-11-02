# Training the gesture classifier

> **You do not need to train anything to play.** Without a trained model the game uses built-in rules (see the main README). Train a model when you want accuracy tailored to your hand, camera and room.

The game recognises three gestures from a 12-number summary of your hand (never from images):

| Gesture | Pose | Notes |
|---|---|---|
| **LEFT** | open palm facing the camera, fingers up, hand tilted so the fingertips point left | tilt is in the image plane, as seen in the mirrored preview |
| **RIGHT** | same, tilted right | |
| **JUMP** | closed fist | any roll |

An upright, relaxed open palm is deliberately *none of these*: the model is not confident there and the game ignores it.

## 1. Record data (about 10 minutes)

```bash
npm install
npm run collect          # opens http://localhost:5173/ml/collect.html
```

1. **Start camera**, pick a gesture, press **Record take**. After a 3-2-1 countdown it captures 60 frames (about 2 s). Keep the pose and move a little.
2. Repeat. **Target per gesture: at least 300 frames and 6 takes**, spread over **3 or more sessions** (reload the page between sessions, ideally at different times of day).
3. Press **Download CSV** after each session and move the files into `ml/data/`.

**What makes data good.** Accuracy on a new player or room is limited by *variety*, not frame count: 60 near-identical frames add little. Change something every take:

- distance (arm's length, closer, farther), position in frame (left, centre, right, high, low)
- tilt strength (shallow, medium, steep: the model must tell "slightly tilted" from "upright")
- right hand *and* left hand, different lighting, a different background
- ideally a second or third person, because hand shapes differ

Mistakes happen: **Undo last take** removes the last one. A take is discarded automatically if your hand leaves the frame for 1.5 s.

The CSV contains 12 numbers per frame plus a take id. No pixels are stored or sent anywhere.

## 2. Train

```bash
python -m venv ml/.venv && source ml/.venv/bin/activate     # Windows: ml\.venv\Scripts\activate
pip install -r ml/requirements.txt
python ml/train.py --data ml/data
```

It splits **by take** (70/15/15; never by row, which would leak near-identical frames into the test set), trains a 12-32-16-3 MLP with light augmentation, and prints test accuracy, per-class precision/recall, a confusion matrix and "gated" accuracy (only frames the app would act on, at its 0.65 confidence threshold). Results are saved to `ml/checkpoints/metrics.json`; copy them into the README metrics table.

The target from the brief is **> 92 % test accuracy**. If you are below it, record more *varied* takes rather than more frames of the same pose.

## 3. Export and install

```bash
python ml/export_onnx.py --data ml/data
```

This folds the input scaling into the first layer, exports to ONNX, applies dynamic INT8 quantisation, verifies it against PyTorch, and writes **`public/models/gesture.int8.onnx`**, which the app loads. Commit that file: CI refuses to deploy without it. Add `--also-fp32` to ship a float model next to it (point `ASSETS.gestureModel` in `src/constants.ts` at it to use it).

Expect the model to be only a few KB, not hundreds: it has about 1,000 parameters. INT8 saves roughly a quarter of that and makes no measurable speed difference at this size; it is kept because the project calls for it, and the script reports the measured numbers.

## 4. Check it in the game

```bash
npm run dev
```

Choose **Use my camera**. On the warm-up screen each gesture card should light up when you make it. Turn on **Settings → Diagnostics** to see hand-model and classifier latency and the camera frame rate: these are the numbers for the latency budget.

## Files

| File | Role |
|---|---|
| `collect.html` + `src/tools/collect.ts` | recorder; reuses the app's own feature code, so training and play can never disagree |
| `dataset.py` | CSV loading, validation, take-level splits; declares the gesture and feature order |
| `train.py` | training and evaluation |
| `export_onnx.py` | ONNX export, INT8 quantisation, verification |
| `test_dataset.py` | `python -m unittest ml/test_dataset.py` |

`GESTURES` and `FEATURE_NAMES` in `dataset.py` must match `src/types.ts` and `src/vision/features.ts`. Every CSV header carries the feature names and `dataset.py` rejects a mismatch, so drift fails loudly.
