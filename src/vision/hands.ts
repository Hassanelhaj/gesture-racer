import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { HandLandmarkerResult } from '@mediapipe/tasks-vision';
import { ASSETS, CAMERA, FEATURES, HAND_TRACKING } from '../constants';
import { createLogger } from '../logger';
import { VisionError } from '../types';
import type { DetectResult } from '../types';

const log = createLogger('vision/hands');

/** A running camera. Call `stop()` to release the device (the green light goes off). */
export interface CameraHandle {
  readonly stream: MediaStream;
  stop: () => void;
}

/**
 * Maps a getUserMedia failure to a VisionError the UI can explain.
 * Browsers report these through DOMException names; anything unrecognised
 * becomes the generic 'camera-failed' so the player still gets a message.
 */
function mapCameraError(cause: unknown): VisionError {
  const name = cause instanceof DOMException ? cause.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
    case 'PermissionDeniedError':
      return new VisionError('camera-denied', 'Camera permission denied', { cause });
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return new VisionError('camera-not-found', 'No suitable camera found', { cause });
    default:
      return new VisionError('camera-failed', 'Camera could not be started', { cause });
  }
}

/**
 * Starts the front camera and attaches it to `video`.
 *
 * The video element is only a decode target for MediaPipe; it never uploads
 * anything. No audio is requested.
 *
 * @param video  element that will receive the stream
 * @param onLost called if the camera disappears mid-game (unplugged, revoked)
 * @throws VisionError with a camera-* code
 */
export async function openCamera(
  video: HTMLVideoElement,
  onLost?: () => void,
): Promise<CameraHandle> {
  // mediaDevices is undefined on insecure origins; the DOM typings say it never is.
  const mediaDevices = (navigator as { mediaDevices?: MediaDevices }).mediaDevices;
  if (mediaDevices === undefined) {
    throw new VisionError('camera-unsupported', 'navigator.mediaDevices is unavailable');
  }

  let stream: MediaStream;
  try {
    stream = await mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: CAMERA.idealWidth },
        height: { ideal: CAMERA.idealHeight },
        frameRate: { ideal: CAMERA.idealFrameRate },
      },
    });
  } catch (cause) {
    throw mapCameraError(cause);
  }

  const stopTracks = (): void => {
    for (const track of stream.getTracks()) track.stop();
  };

  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  try {
    await video.play();
  } catch (cause) {
    stopTracks();
    video.srcObject = null;
    throw new VisionError('camera-failed', 'Video element could not start playback', { cause });
  }

  const handleEnded = (): void => {
    log.warn('camera track ended');
    onLost?.();
  };
  const track = stream.getVideoTracks()[0];
  track?.addEventListener('ended', handleEnded);

  return {
    stream,
    stop: () => {
      track?.removeEventListener('ended', handleEnded);
      stopTracks();
      video.srcObject = null;
    },
  };
}

/** Width / height of the live video, falling back to the default before metadata loads. */
export function videoAspectRatio(video: HTMLVideoElement): number {
  return video.videoWidth > 0 && video.videoHeight > 0
    ? video.videoWidth / video.videoHeight
    : FEATURES.defaultAspectRatio;
}

type Fileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

/** Builds a single-hand VIDEO-mode landmarker on the requested delegate. */
function createLandmarker(fileset: Fileset, delegate: 'GPU' | 'CPU'): Promise<HandLandmarker> {
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: ASSETS.handModel, delegate },
    runningMode: 'VIDEO',
    numHands: HAND_TRACKING.numHands,
    minHandDetectionConfidence: HAND_TRACKING.minDetectionConfidence,
    minHandPresenceConfidence: HAND_TRACKING.minPresenceConfidence,
    minTrackingConfidence: HAND_TRACKING.minTrackingConfidence,
  });
}

/**
 * Thin wrapper over MediaPipe's HandLandmarker.
 *
 * WHY a wrapper: it hides MediaPipe's types behind our own (`DetectResult`),
 * enforces the strictly-increasing-timestamp rule MediaPipe requires, skips
 * duplicate camera frames, and copies landmarks out of MediaPipe-owned memory.
 */
export class HandTracker {
  private readonly landmarker: HandLandmarker;
  /** Which backend actually loaded; useful for the latency budget and bug reports. */
  readonly delegate: 'GPU' | 'CPU';
  private lastVideoTime = -1;
  private lastTimestampMs = Number.NEGATIVE_INFINITY;

  private constructor(landmarker: HandLandmarker, delegate: 'GPU' | 'CPU') {
    this.landmarker = landmarker;
    this.delegate = delegate;
  }

  /**
   * Loads MediaPipe WASM + the hand model. Tries the GPU delegate first and
   * falls back to CPU when the browser/driver cannot provide it, so "no WebGL"
   * degrades speed rather than breaking the game.
   *
   * @throws VisionError('hand-model-failed')
   */
  static async create(): Promise<HandTracker> {
    try {
      const fileset = await FilesetResolver.forVisionTasks(ASSETS.mediapipeWasmDir);
      try {
        const landmarker = await createLandmarker(fileset, 'GPU');
        log.info('hand tracker ready (GPU)');
        return new HandTracker(landmarker, 'GPU');
      } catch (gpuError) {
        log.warn('GPU delegate unavailable, falling back to CPU', gpuError);
        const landmarker = await createLandmarker(fileset, 'CPU');
        log.info('hand tracker ready (CPU)');
        return new HandTracker(landmarker, 'CPU');
      }
    } catch (cause) {
      throw new VisionError('hand-model-failed', 'Could not load the hand-tracking model', {
        cause,
      });
    }
  }

  /**
   * True when the camera has produced a frame this tracker has not yet seen.
   * Cheap and side-effect free, so callers can decide whether to pay for detection.
   */
  hasNewFrame(video: HTMLVideoElement): boolean {
    return (
      video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
      video.videoWidth > 0 &&
      video.currentTime !== this.lastVideoTime
    );
  }

  /** Marks the current frame as seen without analysing it (used when shedding load). */
  skipFrame(video: HTMLVideoElement): void {
    this.lastVideoTime = video.currentTime;
  }

  /**
   * Prepares the tracker for a new camera stream. The trackers are shared across
   * sessions (loading them is the slow part), but a fresh stream restarts
   * `currentTime` at 0, so the "already seen" marker must be cleared. Timestamps
   * keep increasing, which MediaPipe requires.
   */
  reset(): void {
    this.lastVideoTime = -1;
  }

  /**
   * Analyses the current video frame, if it is new.
   *
   * @param video the playing camera element
   * @param nowMs a monotonic clock reading (performance.now())
   * @throws VisionError('tracking-failed') if MediaPipe throws
   */
  detect(video: HTMLVideoElement, nowMs: number): DetectResult {
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth === 0) {
      return { kind: 'stale' };
    }
    // Camera is slower than the render loop: do not re-analyse the same frame.
    if (video.currentTime === this.lastVideoTime) return { kind: 'stale' };
    this.lastVideoTime = video.currentTime;

    // MediaPipe throws on a repeated or decreasing timestamp.
    const timestampMs = nowMs > this.lastTimestampMs ? nowMs : this.lastTimestampMs + 1;
    this.lastTimestampMs = timestampMs;

    let result: HandLandmarkerResult;
    try {
      result = this.landmarker.detectForVideo(video, timestampMs);
    } catch (cause) {
      throw new VisionError('tracking-failed', 'Hand detection failed', { cause });
    }

    const first = result.landmarks[0];
    if (first === undefined) return { kind: 'no-hand' };
    return {
      kind: 'hand',
      frame: {
        // Copy out of MediaPipe-owned objects; drop visibility/presence we do not use.
        landmarks: first.map(({ x, y, z }) => ({ x, y, z })),
        timestampMs,
      },
    };
  }

  /** Frees the MediaPipe graph and WASM memory. */
  close(): void {
    this.landmarker.close();
  }
}
