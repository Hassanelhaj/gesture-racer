# Decisions and assumptions

Each entry: what was decided, why, and what would change it. Section 2 lists **every tunable number** with its rationale and how to validate it; the same text lives next to the value in `src/constants.ts`.

## 1. Decisions

### Architecture

**D1. The game knows nothing about the camera.** `game/` accepts a `Gesture` and a time step. The keyboard, touch and hand tracker all call the same `Game.apply`, and the game can be simulated in tests. *Cost:* one extra layer (`session.ts`).

**D2. `GameSession` is a plain class published through `useSyncExternalStore`.** The 60 fps loop and camera callbacks live outside React; React re-renders on screen changes and at about 10 Hz for numbers. *Alternative rejected:* hooks and refs everywhere, which fought the strict React-hooks lint rules and risked stale closures in the loop.

**D3. One loop, one source of truth for speed.** `Road` owns speed and distance; spawning, score, difficulty and engine pitch derive from it.

**D4. Vision source is an interface (`VisionSource`).** The real pipeline and a scripted fake are interchangeable, which is how the practice screen and the hand-lost pause were tested without a model.

### Vision and ML

**D5. Landmarks become angles and ratios, not raw coordinates.** Curl, spread, openness and roll are invariant to hand size and distance; x and z are scaled by the video aspect ratio so 4:3 and 16:9 cameras agree. *Cost:* a hand pointing straight at the camera has no defined roll, so features return `null` (treated as "no gesture").

**D6. x is mirrored inside `features.ts`.** LEFT means left in the selfie view the player sees. Training uses the same function, so the convention cannot drift.

**D7. The recorder reuses `extractFeatures`; Python never recomputes features.** A second implementation would diverge silently. `dataset.py` also checks every CSV header against its own feature list, so a mismatch fails loudly.

**D8. Three classes plus a confidence gate; no NEUTRAL class.** Per the brief. Resting hands are rejected by `CLASSIFIER.minConfidence = 0.65` (in rules mode, by the tilt dead zone). *Weakest assumption in the project.* If idle hands trigger moves, record a 4th class: add `NEUTRAL` to `GESTURES` (types.ts, dataset.py), record takes, retrain; the game must then ignore it.

**D9. Splits are by take, not by row.** Frames inside a take are near-duplicates; a row split leaks them across train and test and inflates accuracy. At least 3 takes per gesture are required, and the script says so.

**D10. Input normalisation is folded into the first layer at export.** The browser feeds raw features; there is no scaler to ship or keep in sync. The export verifies the fold is exact (max error < 1e-4).

**D11. Export uses MatMul + Add, not Gemm.** ONNX Runtime's dynamic quantiser converts MatMul weights; Gemm would silently stay float.

**D12. INT8 is shipped as the brief requires, with the evidence stated.** On a stand-in model of this architecture: 3,153 B vs 4,315 B, no speed gain, and browser INT8 probabilities differ from Python's by up to 0.022 (top choice identical on all 120 vectors), whereas fp32 matches to 3e-7. *Expected size is therefore a few KB, not the ~200 KB in the brief:* the network has about 1,000 parameters. `--also-fp32` ships a float model; point `ASSETS.gestureModel` at it if you prefer exactness.

**D13. ONNX Runtime uses the WASM backend only, single-threaded.** For this model WebGPU's dispatch and read-back cost more than the compute, and it adds a runtime variant and a failure path. Threads need cross-origin isolation (COOP/COEP headers), which GitHub Pages cannot send. The "WebGPU unsupported" requirement is met by not depending on it; MediaPipe's own GPU delegate falls back to CPU if unavailable.

**D14. ORT's loader and `.wasm` are imported with `?url` and bundled by Vite.** ORT loads its loader with a dynamic `import()`, which Vite's dev server refuses for files in `public/`. (Found by running it in a browser; the first attempt failed.)

**D15. Only MediaPipe's SIMD build is shipped.** Its `_module` and `_nosimd` variants are about 13 MB each and are never requested by this app. Pre-SIMD browsers get the standard model-load error screen.

**D16. Dropped, not queued, inference.** A late gesture is worse than a skipped one (see latency budget).

### Game design

**D17. Discrete lanes with eased movement.** Gestures are coarse and delayed; "one lane per gesture" is forgiving where continuous steering would not be.

**D18. Fairness rules (enforced in `obstacles.ts`, tested).** (a) A row never blocks all three lanes. (b) **Reachability:** for every lane the player might be in after the previous row, the new row leaves a free lane within one step; barrier lanes count as free because a jump clears them. (c) Row spacing is a time gap (1.7 s → 0.95 s) so the reaction window stays similar as speed grows. *Why (b):* the first spawner allowed a two-lane dodge from the road edge, which is physically impossible once reaction time, camera latency and the 200 ms cooldown add up. A latency-limited bot survived only 2 of 20 seeds; with the rule it survives 20 of 20.

**D19. Top speed 580 px/s**, chosen from the latency budget (0.93 s visible warning, about 0.3 s margin).

**D20. Forgiving hitboxes (15 % inset per side)** because sprites have rounded corners and glow.

**D21. The run pauses when the hand is gone for 1.2 s (camera mode).** A helpless car is not fun, and tracking dropouts shorter than a second are ignored.

**D22. The camera is switched off on returning to the menu.** Nothing records while idle; the camera indicator turning off is itself a privacy signal.

**D23. Keyboard and touch fallbacks.** Accessibility, and the only way to play before a model exists. Keys are always active, in camera mode too.

### Platform

**D24. Production-only Content-Security-Policy (`connect-src 'self'`).** Turns "no video leaves the device" into something the browser enforces. Dev is exempt because Vite's HMR needs inline scripts. Verified for ONNX Runtime; **MediaPipe under the CSP is not yet verified**: if it fails, loosen the named directive in `vite.config.ts`.

**D25. Models and WASM are self-hosted** (no CDN), for offline use, the CSP, and so the privacy claim does not depend on a third party.

**D26. Settings in `localStorage`, best score in IndexedDB.** Best score is IndexedDB as specified, with a compare-and-write inside one transaction and an in-memory fallback. Settings are three booleans; `localStorage` is enough.

**D27. Latest majors.** npm resolved React 19, Vite 8, TypeScript 6, ESLint 10 and Vitest 5. All checks pass on them; the lockfile pins them.

**D28. CI warns, but does not fail, when the trained model is missing.** The game degrades to the built-in rules (D29), so a missing file is a quality problem, not an outage; the warning keeps it visible. (An earlier version failed the deploy; that was changed once the fallback existed.)

**D29. Built-in rule-based classifier as a fallback.** *Found by use:* a fresh checkout showed "The gesture model failed to load" because the model is trained from the player's own recordings, so the first-run experience was a dead end. `vision/rules.ts` classifies the same 12 features with fixed rules (open hand + tilt → LEFT/RIGHT, closed hand → JUMP, anything else → none), and `vision/model.ts` falls back to it when the ONNX model cannot load. It is **not silent**: `PipelineStatus.classifier` is `'onnx'` or `'rules'`, shown on the warm-up screen and in Diagnostics. *Alternative rejected:* shipping a model trained on synthetic hands, which would look like the real thing but whose behaviour on real hands I cannot check. *Limits:* the thresholds are geometric guesses, not tuned on real hands; a hand with naturally half-curled fingers may need `RULES.openMaxCurl` raised.

### Performance and loading

**D30. Decorative blurs are gradients, not `filter: blur`.** *Found by measurement:* the page ran at about 26 fps (60 of 63 frames over 20 ms) until the two background glows were removed, then at a flat 60 fps. A CSS blur on screen-sized elements is re-rendered whenever anything on the page repaints, and the canvas repaints every frame. Radial gradients look the same and cost nothing; only `translate` is animated so the movement stays on the compositor. `backdrop-filter` on the menu overlays was removed for the same reason. *Rule for future changes:* no `filter: blur` / `backdrop-filter` over or under the game.

**D31. The gesture model file is fetched before ONNX Runtime is touched, and ONNX Runtime is imported lazily.** A missing model is detected in milliseconds, the built-in rules take over, and the 15 MB runtime is never downloaded or started. The check also rejects an HTML page served with status 200 (dev servers and static hosts do this for unknown paths), which otherwise fails later with a confusing parse error.

**D32. Models load in the background and are shared.** `ModelLoader` starts about 1.2 s after the page opens, when the browser is idle, and is skipped on Save-Data or 2G/3G connections; **Settings → Preload models** turns it off. The tracker and classifier are loaded once and reused when the player returns to the menu and starts again; only the camera is released (D22). *Cost:* a keyboard-only player downloads 22 MB they may never use, which is why it is a setting and why it waits for idle.

**D33. Camera and models load concurrently, with a real progress bar.** A permission prompt no longer delays the download. The loader streams each file once to count bytes (this also fills the HTTP cache, so MediaPipe's own request is served from it) and spends 85 % of the bar on downloading and the rest on start-up. Progress never moves backwards, even when a Content-Length turns out larger than assumed (tested).

**D34. Adaptive tracking instead of a Web Worker, for now.** Detection runs on the main thread. When its average exceeds 14 ms the pipeline analyses every 2nd camera frame and uses a 3-frame vote window (2 agreeing frames at 15 Hz is about the same time as 3 at 30 Hz). It returns to full rate below 9 ms. *Why not a worker:* it is the real fix, but moving MediaPipe into a worker changes how its WebAssembly and GPU delegate load, and I could not test that without a real camera and the hand model; a broken worker would break the game. *Trigger to revisit:* Diagnostics showing Detect above about 15 ms or "Tracking: every 2nd frame" on a normal machine.

**D35. Static scene layers are cached; the canvas is opaque.** Grass, asphalt and edge lines are drawn once into an off-screen canvas with a 24-unit margin (so the crash shake never reveals an edge); the top fade is cached the same way. Each frame draws only what moves. The 2D context is created with `alpha: false`, so the browser does not blend it with the page behind.

## 2. Every tunable number

All live in `src/constants.ts` (and the Python scripts where noted). All are **assumptions chosen before real-world measurement** unless the status column says otherwise.

| Group | Value | Rationale | Validate by |
|---|---|---|---|
| `CAMERA` | 640×480, 30 fps ideal | MediaPipe downsizes internally; more pixels only add cost. The browser may deliver something else. | Diagnostics → camera fps |
| `HAND_TRACKING` detection / presence / tracking confidence | 0.5 each | MediaPipe defaults | Ghost hands appearing → raise |
| `HAND_TRACKING.lostTimeoutMs` | 300 | Short enough to feel instant, long enough not to flicker | Wave a hand out and back |
| `HAND_TRACKING.pauseAfterLostMs` | 1200 | Ignores sub-second dropouts | Briefly cover the camera in a run |
| `FEATURES.dim` | 12 | Contract with the model | n/a |
| `FEATURES.defaultAspectRatio` | 4/3 | Used only before video metadata loads | n/a |
| `FEATURES.palmLengthRef` | 0.15 image heights | Typical palm length at about 50 cm from a laptop webcam | Print `palm_scale` at your normal distance; should sit mid-range |
| `FEATURES.palmScaleMax` / `spreadMax` / `opennessMax` | 2 / 3 / 2.5 | Clamp ceilings that map ratios to 0..1 | Check no feature saturates at 1 for ordinary poses |
| `FEATURES.fingerFoldedAngleRad` | 0.9 (about 52°) | Joint angle regarded as fully curled | Open palm ≈ 0, fist ≈ 1 in the collector's live readout |
| `FEATURES.thumbFoldedAngleRad` | 1.6 (about 92°) | The thumb folds less | Same |
| `FEATURES.minGeometry` | 0.001 image heights | Below this a measurement is degenerate | n/a |
| `CLASSIFIER.minConfidence` | 0.65 | Rejects split LEFT/RIGHT outputs from a resting palm. **Also hard-coded as `APP_MIN_CONFIDENCE` in `ml/train.py`; keep equal.** | Hold the hand upright: confidence should stay below it |
| `CLASSIFIER.inferenceBudgetMs` | 15 | Project brief | Diagnostics → Classify |
| `PRELOAD.startDelayMs` | 1200 | Lets the first paint and title animation finish before a 22 MB download and tracker start-up | Menu animation must not stutter in the first 2 s |
| `PRELOAD.idleTimeoutMs` | 2000 | After the delay, start when idle, or at the latest after this | n/a |
| `PRELOAD.fetchWeight` | 0.85 | Share of the progress bar given to downloading | Bar should not sit at 85 % for long |
| `PRELOAD.expectedBytes` | js 334 kB, wasm 14.47 MB, model 7.8 MB | Sizes for the bar when a server's Content-Length is not trustworthy (compressed responses) | Approximate |
| `ADAPTIVE_TRACKING.slowDetectMs` / `recoverDetectMs` | 14 / 9 ms | About 40 % of a 30 Hz frame; the gap prevents flapping | Diagnostics: "Tracking: every 2nd frame" only on slow machines |
| `ADAPTIVE_TRACKING.minSamples` / `throttledWindow` | 24 / 3 | Ignore the slow warm-up frames; 2 votes at 15 Hz ≈ 3 votes at 30 Hz | A switched gesture still responds in about 150 ms |
| `RULES.tiltStartSin` / `tiltFullSin` | 0.15 / 0.40 | Tilt confidence ramps from about 9° to about 24°; with the 0.65 gate a move registers from about 18°, so a resting hand is ignored | Hold the hand upright and wobble: nothing should fire; tilt 30°: should fire |
| `RULES.openMaxCurl` | 0.5 | Tilt only counts when fingers are mostly straight | Relaxed open hand must stay below it (Diagnostics) |
| `RULES.jumpStartCurl` / `jumpFullCurl` | 0.40 / 0.70 | Fist confidence ramps over this mean-curl range; registers from about 0.60 | Make a fist: should fire; half-close: should not |
| `SMOOTHER.windowSize` | 5 (3 votes needed) | Removes single-frame flicker; about 66 ms at 30 fps | A deliberate tilt responds in under about 150 ms |
| `SMOOTHER.cooldownMs` | 200 | At most 5 actions per second; swallows LEFT↔RIGHT bounce | Sweeping the hand left to right must not double-step |
| `WORLD` | 480×720 | Fixed logical canvas scaled to the screen | n/a |
| `ROAD` | 3 lanes, 360 wide, dashes 44/36, edge 4 | 120 px lanes leave a 60 px shoulder | n/a |
| `SPEED` | 260 → 580 px/s, +10 px/s² | Latency budget, section 3; ceiling reached in about 32 s | Play 60 s; harder, never abrupt |
| `CAR.width/height/y` | 56 × 96, top at 540 | Fixed screen position | n/a |
| `CAR.laneChangeRate` | 18 /s | 95 % of a lane change in 0.17 s | Matches the 200 ms gesture cadence |
| `CAR.jumpDurationS` | 0.8 | About 0.58 s of airborne safe window | Jump at start speed: window must feel forgiving |
| `CAR.airborneThreshold` | 0.3 of peak | Height at which a barrier is cleared | Same |
| `CAR.jumpPeakScale` / `jumpLiftPx` | 1.3 / 36 | Visual only | n/a |
| `OBSTACLES` car / barrier size | 56 × 96 / 96 × 34 | Barrier narrower than a 120 px lane | n/a |
| `OBSTACLES.firstRowDistancePx` | 700 | 2.7 s of grace at start speed | n/a |
| `OBSTACLES.gapSecondsStart` / `gapSecondsMin` | 1.7 / 0.95 | One lane change + cooldown + recovery | The bot test and play-testing |
| `OBSTACLES.barrierChance` | 0.35 | A third of obstacles can be jumped | Variety in play |
| `OBSTACLES.doubleBlockChanceMax` | 0.45 | Chance of a 2-lane row at full difficulty | Difficulty feel |
| `OBSTACLES.despawnMarginPx` | 120 | Cull after leaving the screen | n/a |
| `COLLISION.hitboxTolerance` | 0.15 per side | About 8 px sideways and 14 px front/back on a car | Brushing past a car should feel fair |
| `SCORE.pxPerMeter` / `passBonus` | 25 / 10 | 260–580 px/s reads as about 37–84 km/h | n/a |
| `LOOP.maxDeltaS` | 0.05 | 20 fps floor; prevents teleporting after a stall | n/a |
| `FEEDBACK` | shake 0.5 s / 14 px, flash 0.35 s | Crash feel | n/a |
| `AUDIO` | master 0.25, engine 0.1, 55–150 Hz, filter 300–900 Hz, crash 0.6 s | Quiet by default | Listen; I could not |
| `UI.refreshMs` | 100 | 10 Hz reads smooth and keeps React off the 60 fps path | n/a |
| `UI.gameOverDelayMs` | 750 | Let the crash animation play | n/a |
| `UI.tapZoneFraction` | 1/3 | Outer thirds steer, middle jumps | Touch play-test |
| Recorder `TAKE_FRAMES` / `MAX_HAND_GAP_MS` / countdown | 60 / 1500 ms / 3 s | About 2 s per take; discard a take if the hand vanishes | n/a |
| `train.py` network, lr, weight decay, patience | 12-32-16-3, 3e-3, 1e-4, 60 | Small model; early stopping on validation loss | Training curves |
| `train.py` `NOISE_STD` / `ROLL_JITTER_RAD` | 0.02 / 0.06 | Mimics tracker jitter and slight pose variation | Test accuracy with and without |
| `train.py` split | 70 / 15 / 15 by take | Standard | n/a |
| `train.py` `TARGET_ACCURACY` | 0.92 | Project brief | n/a |
| `export_onnx.py` opset, tolerances | 17, 1e-4 fp32 error, 0.99 INT8 agreement | Conservative checks | n/a |

## 3. Known gaps

1. MediaPipe has never run in this project's tests (no model download in the sandbox).
2. The built-in rules have been tested only on synthetic hands built from the same geometry as the features, not on real ones.
3. The performance numbers come from headless Chromium with software rendering. Real GPUs, real hand-detection cost (MediaPipe never ran in my tests) and Firefox/Safari are unmeasured; the blur finding is a cause isolated on one machine and is expected, not proven, to carry over.
4. `train.py` and the PyTorch half of `export_onnx.py` have never executed (PyTorch not installable there); they were reviewed by hand and the INT8 half was verified separately.
5. No real-hand accuracy or end-to-end latency has been measured.
6. Firefox, Safari, iOS and real GPUs are untested.
7. The human-reaction figure (0.25 s) in the latency budget is an assumption.
