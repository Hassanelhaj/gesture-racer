/**
 * Every tunable threshold in the project, grouped by subsystem.
 *
 * RULE: no magic numbers elsewhere. Each value below is an ASSUMPTION chosen
 * before real-world measurement; docs/decisions.md records why, and the
 * "validate with" note says how to tune it. Game-layer constants are appended
 * to this file in Step 3.
 */

const BASE = import.meta.env.BASE_URL;

/**
 * Where self-hosted runtime assets live (copied/fetched by scripts/setup-assets.mjs).
 * The ONNX Runtime WASM is deliberately absent: Vite bundles it as a hashed
 * asset via `?url` imports in vision/classifier.ts.
 */
export const ASSETS = {
  handModel: `${BASE}models/hand_landmarker.task`,
  gestureModel: `${BASE}models/gesture.int8.onnx`,
  mediapipeWasmDir: `${BASE}wasm/mediapipe`,
} as const;

/**
 * Camera request. 640x480 @ 30 fps: MediaPipe downsizes its input internally,
 * so higher resolution only adds decode/upload cost. These are "ideal" hints;
 * the browser may deliver something else.
 */
export const CAMERA = {
  idealWidth: 640,
  idealHeight: 480,
  idealFrameRate: 30,
} as const;

/**
 * MediaPipe HandLandmarker settings.
 * 0.5 values are MediaPipe's own defaults; raise them if ghost hands appear.
 * lostTimeoutMs: how long with no hand before the game treats the hand as
 * gone (pause hint, steering released). Validate with: wave a hand out of
 * frame and back; 300 ms should feel instant but not flicker.
 * pauseAfterLostMs: with the camera as the only input, a missing hand means a
 * helpless car, so the run freezes after this long and resumes when the hand
 * returns (assumption: long enough to ignore a one-second tracking dropout).
 */
export const HAND_TRACKING = {
  numHands: 1,
  minDetectionConfidence: 0.5,
  minPresenceConfidence: 0.5,
  minTrackingConfidence: 0.5,
  lostTimeoutMs: 300,
  pauseAfterLostMs: 1200,
} as const;

/**
 * Feature extraction. See vision/features.ts for the meaning of each feature.
 *
 * dim                  : length of the classifier input vector.
 * defaultAspectRatio   : used only when the video size is not yet known.
 * palmLengthRef        : typical wrist->middle-knuckle length in image-height
 *                        units at ~50 cm from a laptop webcam (assumption).
 * palmScaleMax         : palm scale is clamped to palmLengthRef * this, then mapped to 0..1.
 * spreadMax / opennessMax : clamp ceilings that map those ratios to 0..1.
 * fingerFoldedAngleRad : joint angle (at the middle finger joint) regarded as
 *                        "fully curled" for the four fingers (~52 degrees).
 * thumbFoldedAngleRad  : same for the thumb, which folds less (~92 degrees).
 * minGeometry          : below this length (image-height units) a hand
 *                        measurement is treated as degenerate (e.g. palm
 *                        pointing straight at the camera).
 * Validate with: print features for an open palm and a fist; curls should sit
 * near 0 and near 1 respectively.
 */
export const FEATURES = {
  dim: 12,
  defaultAspectRatio: 4 / 3,
  palmLengthRef: 0.15,
  palmScaleMax: 2,
  spreadMax: 3,
  opennessMax: 2.5,
  fingerFoldedAngleRad: 0.9,
  thumbFoldedAngleRad: 1.6,
  minGeometry: 1e-3,
} as const;

/**
 * Classifier.
 * minConfidence: below this softmax probability the output is "no gesture".
 * The model has only LEFT/RIGHT/JUMP classes, so an upright neutral palm must
 * be rejected by low confidence rather than by a NEUTRAL class (assumption).
 * Validate with: hold the hand upright; confidence should stay under 0.65.
 * inferenceBudgetMs: target from the project brief (<15 ms).
 */
export const CLASSIFIER = {
  minConfidence: 0.65,
  inferenceBudgetMs: 15,
} as const;

/**
 * Adaptive tracking load. Hand detection runs on the main thread, so on a slow
 * machine (CPU delegate, integrated GPU) it can eat most of a 60 fps frame and
 * make the whole game stutter. When the measured detection time stays above
 * slowDetectMs, the pipeline analyses every 2nd camera frame instead of every
 * frame (halving that cost) and shortens the smoother's vote window to
 * throttledWindow so the response stays about as quick. It returns to full rate
 * once detection is faster than recoverDetectMs (the gap prevents flapping).
 * minSamples: processed frames to average before changing mode.
 * Validate with: Settings > Diagnostics shows "Tracking: every 2nd frame" on a slow machine.
 */
export const ADAPTIVE_TRACKING = {
  slowDetectMs: 14,
  recoverDetectMs: 9,
  minSamples: 24,
  throttledWindow: 3,
} as const;

/**
 * Model download and preloading.
 * fetchWeight   : share of the loading bar given to downloading; the rest is
 *                 compiling and starting the models.
 * startDelayMs  : preloading waits this long after the page is up, so the first
 *                 paint and the title animation are never competing with a 22 MB
 *                 download and the hand tracker's start-up.
 * idleTimeoutMs : after that delay it starts when the browser is idle, or after
 *                 this long at the latest.
 * expectedBytes : download sizes used for the progress bar when the server does
 *                 not report a trustworthy length (compressed responses).
 *                 Approximate (MediaPipe WASM JS/WASM, hand model).
 */
export const PRELOAD = {
  fetchWeight: 0.85,
  startDelayMs: 1200,
  idleTimeoutMs: 2000,
  expectedBytes: { js: 334_000, wasm: 14_470_000, model: 7_800_000, other: 1_000_000 },
} as const;

/**
 * Built-in rule-based classifier, used when no trained model is installed.
 * It reads only two things: how far the hand is tilted (|roll_sin|) and how
 * curled the four fingers are (mean curl, thumb excluded). All values are
 * assumptions to validate by trying the gestures with Settings > Diagnostics on.
 *
 * tiltStartSin / tiltFullSin : tilt confidence rises linearly from 0 at
 *      sin 0.15 (about 9 degrees) to 1 at sin 0.40 (about 24 degrees). With
 *      CLASSIFIER.minConfidence = 0.65 a move registers from about 18 degrees,
 *      so a resting hand (0-10 degrees) is ignored: that dead zone is the
 *      rule-based stand-in for a NEUTRAL class.
 * openMaxCurl : tilt only counts for an open hand (mean curl below this).
 * jumpStartCurl / jumpFullCurl : fist confidence rises from 0 at mean curl 0.40
 *      to 1 at 0.70; with the 0.65 gate a fist registers from curl ~0.60.
 *      Curls between openMaxCurl and that point are deliberately ambiguous
 *      (half-closed hand) and produce no gesture.
 */
export const RULES = {
  tiltStartSin: 0.15,
  tiltFullSin: 0.4,
  openMaxCurl: 0.5,
  jumpStartCurl: 0.4,
  jumpFullCurl: 0.7,
} as const;

/**
 * Temporal smoothing. 5-frame majority vote (3 agreeing frames needed, about
 * 100 ms at 30 fps) and a 200 ms cooldown between emitted gestures, i.e. at
 * most 5 actions per second. Validate with: a single mis-classified frame must
 * never move the car; a deliberate tilt should respond in under ~150 ms.
 */
export const SMOOTHER = {
  windowSize: 5,
  cooldownMs: 200,
} as const;

/* ------------------------------------------------------------------------- *
 * GAME LAYER (Step 3)
 * All lengths are in "world pixels": a fixed 480x720 logical canvas that
 * render.ts scales to the real canvas. Speeds are world px/second.
 * ------------------------------------------------------------------------- */

/** Logical canvas size (2:3 portrait). The real canvas is scaled to fit it. */
export const WORLD = {
  width: 480,
  height: 720,
} as const;

/**
 * Road geometry. 3 lanes of 120 px leave a 60 px shoulder each side.
 * dashLength + dashGap is the lane-marking period; road.ts scrolls by it.
 */
export const ROAD = {
  laneCount: 3,
  width: 360,
  dashLength: 44,
  dashGap: 36,
  edgeLineWidth: 4,
} as const;

/**
 * Speed ramp (assumption). 260 px/s is slow enough to read the first obstacle.
 * 580 px/s is the ceiling: a row becomes visible ~0.93 s before impact
 * (540 px of road above the car), and a one-lane dodge needs roughly
 * 0.4 s (human reaction + camera + 3-frame smoother vote) + 0.2 s (lane change)
 * of that. Faster than this the margin gets uncomfortably thin.
 * acceleration 10 px/s^2 reaches the ceiling in ~32 s.
 * Validate with: play 60 s; it should feel steadily harder, never abruptly.
 */
export const SPEED = {
  start: 260,
  max: 580,
  acceleration: 10,
} as const;

/**
 * Player car.
 * laneChangeRate : exponential easing rate (1/s); 18 closes ~95% of a lane
 *                  change in ~0.17 s, fast enough for a 200 ms gesture cadence.
 * jumpDurationS  : total airtime. The car counts as "airborne" (can clear
 *                  barriers) while its height is >= airborneThreshold of the peak,
 *                  i.e. ~72% of the jump = ~0.58 s of safe window.
 * jumpPeakScale / jumpLiftPx : purely visual (fake height in a top-down view).
 * y              : top edge of the car; it never moves vertically in world space.
 */
export const CAR = {
  width: 56,
  height: 96,
  y: 540,
  laneChangeRate: 18,
  jumpDurationS: 0.8,
  airborneThreshold: 0.3,
  jumpPeakScale: 1.3,
  jumpLiftPx: 36,
} as const;

/**
 * Obstacles and spawning.
 *
 * FAIRNESS RULES (enforced in obstacles.ts, verified by tests):
 *  - rows never block all three lanes (at most 2 of 3 get an obstacle);
 *  - REACHABILITY: a free lane must be within one lane of wherever the player
 *    may be standing, so no row demands a two-lane dodge;
 *  - rows are spaced by a TIME gap that shrinks from gapSecondsStart to
 *    gapSecondsMin as speed ramps. 0.95 s covers: one lane change + the
 *    smoother's 200 ms cooldown + recovering to a useful lane before the next row.
 *    Spacing in pixels = gap seconds * current speed.
 *
 * firstRowDistancePx : grace distance before the first row.
 * barrierChance      : probability an occupied lane holds a jumpable barrier
 *                      instead of a car.
 * doubleBlockChanceMax : at full difficulty, chance a row occupies 2 lanes.
 */
export const OBSTACLES = {
  carWidth: 56,
  carHeight: 96,
  barrierWidth: 96,
  barrierHeight: 34,
  carVariants: 5,
  firstRowDistancePx: 700,
  gapSecondsStart: 1.7,
  gapSecondsMin: 0.95,
  barrierChance: 0.35,
  doubleBlockChanceMax: 0.45,
  despawnMarginPx: 120,
} as const;

/**
 * Collision forgiveness. Each hitbox is shrunk by this fraction of its own
 * width/height on EVERY side before the overlap test, so clipping a bumper
 * does not end the run. 0.15 on a 56x96 car forgives ~8 px sideways and
 * ~14 px front/back. Validate with: brushing past a car should feel fair.
 */
export const COLLISION = {
  hitboxTolerance: 0.15,
} as const;

/**
 * Scoring. 25 px = 1 m, so 260..580 px/s is ~37..84 km/h (HUD speed).
 * passBonus is awarded when an obstacle row is fully passed.
 */
export const SCORE = {
  pxPerMeter: 25,
  passBonus: 10,
} as const;

/**
 * Game loop. Frame delta is clamped so a backgrounded tab or a GC pause cannot
 * teleport the car through obstacles in one giant step (50 ms = 20 fps floor).
 */
export const LOOP = {
  maxDeltaS: 0.05,
} as const;

/** Crash feedback durations and strength. */
export const FEEDBACK = {
  crashShakeS: 0.5,
  crashShakePx: 14,
  crashFlashS: 0.35,
} as const;

/**
 * Audio levels (linear gain) and engine pitch range. masterGain is kept low
 * on purpose: nobody wants a loud surprise when the page first makes sound.
 */
export const AUDIO = {
  masterGain: 0.25,
  engineGain: 0.1,
  engineMinHz: 55,
  engineMaxHz: 150,
  engineFilterMinHz: 300,
  engineFilterMaxHz: 900,
  crashNoiseS: 0.6,
} as const;

/**
 * UI behaviour.
 * refreshMs        : how often React re-reads fast-changing numbers (score,
 *                    vision status). 10 Hz reads as smooth and keeps React off
 *                    the 60 fps render path.
 * gameOverDelayMs  : crash animation plays this long before the game-over card.
 * tapZoneFraction  : touch fallback; the outer thirds of the screen steer,
 *                    the middle third jumps.
 */
export const UI = {
  refreshMs: 100,
  gameOverDelayMs: 750,
  tapZoneFraction: 1 / 3,
} as const;
