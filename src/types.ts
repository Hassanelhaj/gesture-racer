/**
 * Shared domain types. Kept free of runtime dependencies (apart from the small
 * VisionError class) so every layer, including tests, can import it cheaply.
 */

/**
 * The gestures the classifier can emit.
 *
 * ORDER IS A CONTRACT: index 0 = LEFT, 1 = RIGHT, 2 = JUMP. It must match the
 * class indices used by ml/train.py and the output logits of the ONNX model.
 */
export const GESTURES = ['LEFT', 'RIGHT', 'JUMP'] as const;
export type Gesture = (typeof GESTURES)[number];

/** A single hand landmark in MediaPipe's normalized image space (x, y in 0..1). */
export interface Landmark {
  readonly x: number;
  readonly y: number;
  /** Relative depth; roughly the same scale as x. Noisy, used sparingly. */
  readonly z: number;
}

/** The 21 MediaPipe hand landmarks, index 0 = wrist. */
export type HandLandmarks = readonly Landmark[];

/** Classifier input: FEATURES.dim floats, see FEATURE_NAMES in vision/features.ts. */
export type FeatureVector = Float32Array;

/** One tracked hand observed in one camera frame. */
export interface HandFrame {
  readonly landmarks: HandLandmarks;
  /** Monotonic timestamp (ms) passed to MediaPipe for this frame. */
  readonly timestampMs: number;
}

/**
 * Outcome of asking the tracker about the current video frame.
 *  - stale   : the camera has not produced a new frame since last time; do nothing.
 *  - no-hand : a new frame was analysed and no hand is visible.
 *  - hand    : a new frame was analysed and a hand was found.
 * The three-way split lets the game count "hand lost" time only on real frames.
 */
export type DetectResult =
  | { readonly kind: 'stale' }
  | { readonly kind: 'no-hand' }
  | { readonly kind: 'hand'; readonly frame: HandFrame };

/** Result of one classifier inference. */
export interface GesturePrediction {
  /** The winning gesture, or null when confidence is below CLASSIFIER.minConfidence. */
  readonly gesture: Gesture | null;
  /** Softmax probability of the most likely class (even when gated to null). */
  readonly confidence: number;
  readonly probabilities: Readonly<Record<Gesture, number>>;
  /** Wall-clock time of the ONNX run, for the latency budget. */
  readonly inferenceMs: number;
}

/** Everything that can go wrong in the vision pipeline, as a closed set the UI can switch on. */
export type VisionErrorCode =
  | 'camera-unsupported'
  | 'camera-denied'
  | 'camera-not-found'
  | 'camera-failed'
  | 'camera-lost'
  | 'hand-model-failed'
  | 'tracking-failed'
  | 'classifier-load-failed'
  | 'classifier-inference-failed';

/** Error carrying a machine-readable code; `cause` keeps the original browser/library error. */
export class VisionError extends Error {
  readonly code: VisionErrorCode;

  constructor(code: VisionErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'VisionError';
    this.code = code;
  }
}

/**
 * Player-facing text for each error code. Lives next to the codes so a new
 * code cannot be added without also deciding what the player is told.
 */
export const VISION_ERROR_MESSAGES: Readonly<Record<VisionErrorCode, string>> = {
  'camera-unsupported':
    'This browser cannot access a camera here. Use a recent Chrome, Edge, Firefox or Safari over HTTPS.',
  'camera-denied':
    'Camera access was blocked. Allow the camera for this site in your browser settings, then reload. Video never leaves your device.',
  'camera-not-found': 'No camera was found. Connect a webcam and reload.',
  'camera-failed':
    'The camera could not be started. Another app may be using it. Close it and try again.',
  'camera-lost': 'The camera stopped working. Check the connection and reload.',
  'hand-model-failed':
    'The hand-tracking model failed to load. Check your connection and reload.',
  'tracking-failed': 'Hand tracking stopped unexpectedly. Reload the page to try again.',
  'classifier-load-failed':
    'The gesture model failed to load. Check your connection and reload.',
  'classifier-inference-failed': 'Gesture recognition stopped unexpectedly. Reload the page.',
};
