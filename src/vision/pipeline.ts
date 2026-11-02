import { ADAPTIVE_TRACKING } from '../constants';
import { createLogger } from '../logger';
import { VisionError } from '../types';
import type { DetectResult, Gesture } from '../types';
import type { GestureModel } from './classifier';
import { extractFeatures } from './features';
import { openCamera, videoAspectRatio } from './hands';
import type { CameraHandle, HandTracker } from './hands';
import { visionModels } from './loader';
import type { ModelLoader } from './loader';
import { GestureSmoother } from './smoother';

const log = createLogger('vision/pipeline');

/** Live numbers for the HUD and the diagnostics panel. */
export interface PipelineStatus {
  readonly handPresent: boolean;
  /** Classifier output for the latest frame (before smoothing). */
  readonly rawGesture: Gesture | null;
  /** Smoothed, stable gesture. */
  readonly stableGesture: Gesture | null;
  readonly confidence: number;
  readonly detectMs: number;
  readonly inferenceMs: number;
  readonly cameraFps: number;
  readonly delegate: 'GPU' | 'CPU';
  /** Which classifier is running: the trained ONNX model or the built-in rules. */
  readonly classifier: GestureModel['kind'];
  /** True while load shedding is on: only every 2nd camera frame is analysed. */
  readonly throttled: boolean;
}

export interface PipelineEvents {
  /** A new stable gesture was established (fires once per transition, never for "none"). */
  readonly onGesture: (gesture: Gesture) => void;
  /** Unrecoverable failure mid-run (camera unplugged, tracker crashed). */
  readonly onError: (error: VisionError) => void;
}

/**
 * What the game session needs from a vision source. The real implementation
 * is `VisionPipeline`; tests and demos can substitute a fake.
 */
export interface VisionSource {
  /** Processes the newest camera frame; call once per render frame. */
  step: (nowMs: number) => void;
  readonly status: PipelineStatus;
  /** How long no hand has been visible, in ms (0 while one is). */
  handLostMs: (nowMs: number) => number;
  /** Clears smoothing state, e.g. at the start of a run. */
  reset: () => void;
  /** Releases the camera. The shared models stay loaded for the next session. */
  dispose: () => Promise<void>;
}

/** `fraction` is 0..1, or null when progress cannot be measured. */
export type ProgressCallback = (message: string, fraction: number | null) => void;

export type VisionFactory = (
  video: HTMLVideoElement,
  events: PipelineEvents,
  onProgress: ProgressCallback,
) => Promise<VisionSource>;

const EMA_ALPHA = 0.1;
const ema = (previous: number, sample: number): number =>
  previous === 0 ? sample : previous + (sample - previous) * EMA_ALPHA;

/**
 * Camera -> MediaPipe -> features -> classifier -> smoother, as one unit.
 *
 * Flow control, in two layers:
 *  - if the classifier is still busy when a frame arrives, that frame is dropped
 *    (a gesture recognised late is worse than one skipped);
 *  - if hand detection itself is slow on this machine, every 2nd camera frame is
 *    skipped (see ADAPTIVE_TRACKING), trading gesture rate for a smooth game.
 *
 * The hand tracker and classifier are shared (see ModelLoader); this class owns
 * only the camera and the per-session state.
 */
export class VisionPipeline implements VisionSource {
  private smoother: GestureSmoother;
  private readonly normalSmoother = new GestureSmoother();
  private readonly throttledSmoother = new GestureSmoother({ windowSize: ADAPTIVE_TRACKING.throttledWindow });
  private disposed = false;
  private failed = false;
  private handLostSinceMs: number | null = null;
  private lastFrameMs: number | null = null;

  private handPresent = false;
  private rawGesture: Gesture | null = null;
  private stableGesture: Gesture | null = null;
  private confidence = 0;
  private detectMs = 0;
  private inferenceMs = 0;
  private frameIntervalMs = 0;

  private throttled = false;
  private skipThisFrame = false;
  private samplesInMode = 0;

  private constructor(
    private readonly video: HTMLVideoElement,
    private readonly camera: CameraHandle,
    private readonly tracker: HandTracker,
    private readonly classifier: GestureModel,
    private readonly events: PipelineEvents,
  ) {
    this.smoother = this.normalSmoother;
    this.tracker.reset();
  }

  /**
   * Opens the camera and waits for the shared models, concurrently: the browser's
   * permission prompt no longer delays the model download (or the other way
   * round). A permission denial is reported first, immediately.
   *
   * @throws VisionError
   */
  static async create(
    video: HTMLVideoElement,
    events: PipelineEvents,
    onProgress: ProgressCallback = () => undefined,
    loader: ModelLoader = visionModels,
  ): Promise<VisionPipeline> {
    let cameraReady = false;
    const unsubscribe = loader.subscribe((progress) => {
      onProgress(cameraReady ? progress.message : 'Waiting for camera permission…', progress.fraction);
    });

    const cameraPromise = openCamera(video, () => {
      events.onError(new VisionError('camera-lost', 'Camera disconnected'));
    });
    const modelsPromise = loader.load();
    try {
      let camera: CameraHandle;
      try {
        camera = await cameraPromise;
      } catch (error) {
        modelsPromise.catch(() => undefined); // keep loading for a retry; just do not report it twice
        throw error;
      }
      cameraReady = true;
      onProgress('Loading models…', null);
      let models;
      try {
        models = await modelsPromise;
      } catch (error) {
        camera.stop();
        throw error instanceof VisionError
          ? error
          : new VisionError('hand-model-failed', 'Could not load the hand-tracking model', { cause: error });
      }
      return new VisionPipeline(video, camera, models.tracker, models.classifier, events);
    } finally {
      unsubscribe();
    }
  }

  get status(): PipelineStatus {
    return {
      handPresent: this.handPresent,
      rawGesture: this.rawGesture,
      stableGesture: this.stableGesture,
      confidence: this.confidence,
      detectMs: this.detectMs,
      inferenceMs: this.inferenceMs,
      cameraFps: this.frameIntervalMs > 0 ? 1000 / this.frameIntervalMs : 0,
      delegate: this.tracker.delegate,
      classifier: this.classifier.kind,
      throttled: this.throttled,
    };
  }

  handLostMs(nowMs: number): number {
    return this.handLostSinceMs === null ? 0 : Math.max(0, nowMs - this.handLostSinceMs);
  }

  step(nowMs: number): void {
    if (this.disposed || this.failed) return;
    if (!this.tracker.hasNewFrame(this.video)) return;

    // Load shedding: consume every 2nd frame without analysing it.
    if (this.throttled) {
      this.skipThisFrame = !this.skipThisFrame;
      if (this.skipThisFrame) {
        this.tracker.skipFrame(this.video);
        return;
      }
    }

    const startedAt = performance.now();
    let result: DetectResult;
    try {
      result = this.tracker.detect(this.video, nowMs);
    } catch (error) {
      this.fail(error instanceof VisionError ? error : new VisionError('tracking-failed', 'Tracking failed', { cause: error }));
      return;
    }
    if (result.kind === 'stale') return;

    this.detectMs = ema(this.detectMs, performance.now() - startedAt);
    this.adaptToLoad();
    if (this.lastFrameMs !== null) {
      // Interval between ANALYSED frames; while throttled, the camera really runs at twice this rate.
      const interval = (nowMs - this.lastFrameMs) / (this.throttled ? 2 : 1);
      this.frameIntervalMs = ema(this.frameIntervalMs, interval);
    }
    this.lastFrameMs = nowMs;

    if (result.kind === 'no-hand') {
      this.handLostSinceMs ??= nowMs;
      this.handPresent = false;
      this.confidence = 0;
      this.feed(null, nowMs);
      return;
    }

    this.handLostSinceMs = null;
    this.handPresent = true;
    const features = extractFeatures(result.frame.landmarks, videoAspectRatio(this.video));
    if (features === null) {
      this.feed(null, nowMs);
      return;
    }
    if (this.classifier.busy) return; // drop the frame rather than queue it

    this.classifier
      .predict(features)
      .then((prediction) => {
        if (this.disposed) return;
        this.inferenceMs = ema(this.inferenceMs, prediction.inferenceMs);
        this.confidence = prediction.confidence;
        this.feed(prediction.gesture, nowMs);
      })
      .catch((error: unknown) => {
        this.fail(
          error instanceof VisionError
            ? error
            : new VisionError('classifier-inference-failed', 'Inference failed', { cause: error }),
        );
      });
  }

  reset(): void {
    this.smoother.reset();
    this.rawGesture = null;
    this.stableGesture = null;
  }

  dispose(): Promise<void> {
    if (!this.disposed) {
      this.disposed = true;
      this.camera.stop(); // models are shared and stay loaded
    }
    return Promise.resolve();
  }

  /** Switches load shedding on or off based on how long detection takes. */
  private adaptToLoad(): void {
    this.samplesInMode += 1;
    if (this.samplesInMode < ADAPTIVE_TRACKING.minSamples) return;
    const shouldThrottle = this.throttled
      ? this.detectMs > ADAPTIVE_TRACKING.recoverDetectMs
      : this.detectMs > ADAPTIVE_TRACKING.slowDetectMs;
    if (shouldThrottle === this.throttled) return;

    this.throttled = shouldThrottle;
    this.samplesInMode = 0;
    this.skipThisFrame = false;
    this.smoother = shouldThrottle ? this.throttledSmoother : this.normalSmoother;
    this.smoother.reset();
    log.info(shouldThrottle ? 'detection is slow; analysing every 2nd frame' : 'detection recovered; analysing every frame', {
      detectMs: this.detectMs,
    });
  }

  private feed(observation: Gesture | null, nowMs: number): void {
    this.rawGesture = observation;
    const output = this.smoother.update(observation, nowMs);
    this.stableGesture = output.gesture;
    if (output.changed && output.gesture !== null) this.events.onGesture(output.gesture);
  }

  private fail(error: VisionError): void {
    if (this.failed || this.disposed) return;
    this.failed = true;
    log.error('vision pipeline failed', error);
    this.events.onError(error);
  }
}

/** Default factory used by the app. */
export const createVisionPipeline: VisionFactory = (video, events, onProgress) =>
  VisionPipeline.create(video, events, onProgress);
