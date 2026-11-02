import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ADAPTIVE_TRACKING } from '../src/constants';
import { VisionError } from '../src/types';
import type { DetectResult, Gesture } from '../src/types';
import type { HandTracker } from '../src/vision/hands';
import { ModelLoader } from '../src/vision/loader';
import { VisionPipeline } from '../src/vision/pipeline';
import { RuleBasedClassifier } from '../src/vision/rules';
import { makeHand } from './helpers/makeHand';

const mocks = vi.hoisted(() => {
  const initial = (): DetectResult => ({ kind: 'no-hand' });
  return {
    result: initial(),
    stopCamera: vi.fn(),
    openCamera: vi.fn(),
    closeTracker: vi.fn(),
    skipFrame: vi.fn(),
    detect: vi.fn(),
    createTracker: vi.fn(),
    /** Fake clock: `detect` advances it, so "how long detection took" is under test control. */
    clock: { now: 0, detectCostMs: 0 },
  };
});

// Only the camera is replaced here; the hand tracker comes from the injected loader below.
vi.mock('../src/vision/hands', () => ({
  openCamera: mocks.openCamera,
  videoAspectRatio: () => 4 / 3,
  HandTracker: {},
}));

const video = {} as HTMLVideoElement;
const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));
const handShowing = (roll: number, curl = 0.1): DetectResult => ({
  kind: 'hand',
  frame: { landmarks: makeHand({ roll, curl, aspect: 4 / 3 }), timestampMs: 0 },
});

function fakeTracker(): HandTracker {
  return {
    delegate: 'CPU',
    hasNewFrame: () => mocks.result.kind !== 'stale',
    skipFrame: mocks.skipFrame,
    reset: vi.fn(),
    close: mocks.closeTracker,
    detect: () => {
      mocks.detect();
      mocks.clock.now += mocks.clock.detectCostMs;
      return mocks.result;
    },
  } as unknown as HandTracker;
}

/** A loader whose "downloads" are instant and whose classifier is the built-in rules. */
function makeLoader(): ModelLoader {
  mocks.createTracker.mockImplementation(() => Promise.resolve(fakeTracker()));
  return new ModelLoader({
    prefetch: (_urls, report) => {
      report(10, 20);
      report(20, 20);
      return Promise.resolve();
    },
    createTracker: mocks.createTracker as () => Promise<HandTracker>,
    loadClassifier: () => Promise.resolve(new RuleBasedClassifier()),
  });
}

async function createPipeline(loader = makeLoader()) {
  const gestures: Gesture[] = [];
  const errors: unknown[] = [];
  const progress: { message: string; fraction: number | null }[] = [];
  const pipeline = await VisionPipeline.create(
    video,
    { onGesture: (g) => gestures.push(g), onError: (e) => errors.push(e) },
    (message, fraction) => progress.push({ message, fraction }),
    loader,
  );
  return { pipeline, gestures, errors, progress, loader };
}

/** Runs `frames` camera frames, 33 ms apart, waiting for each async classification. */
async function run(pipeline: VisionPipeline, frames: number, startMs = 1000): Promise<void> {
  for (let i = 0; i < frames; i += 1) {
    pipeline.step(startMs + i * 33);
    await flush();
  }
}

beforeEach(() => {
  mocks.result = { kind: 'no-hand' };
  mocks.clock.now = 0;
  mocks.clock.detectCostMs = 0;
  for (const fn of [mocks.stopCamera, mocks.closeTracker, mocks.skipFrame, mocks.detect, mocks.createTracker, mocks.openCamera]) fn.mockReset();
  mocks.openCamera.mockImplementation(() => Promise.resolve({ stream: {}, stop: mocks.stopCamera }));
  vi.spyOn(performance, 'now').mockImplementation(() => mocks.clock.now);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('VisionPipeline with the built-in rules (no trained model)', () => {
  it('starts, and says it is using the rules', async () => {
    const { pipeline } = await createPipeline();
    expect(pipeline.status.classifier).toBe('rules');
    expect(pipeline.status.delegate).toBe('CPU');
    expect(pipeline.status.throttled).toBe(false);
  });

  it('turns a held right tilt into exactly one RIGHT after 3 agreeing frames', async () => {
    const { pipeline, gestures } = await createPipeline();
    mocks.result = handShowing(0.8);
    await run(pipeline, 2);
    expect(gestures).toEqual([]); // 2 of 5 votes is not a majority
    await run(pipeline, 6, 1100);
    expect(gestures).toEqual(['RIGHT']);
    expect(pipeline.status.stableGesture).toBe('RIGHT');
  });

  it('turns a fist into JUMP and a left tilt into LEFT', async () => {
    const { pipeline, gestures } = await createPipeline();
    mocks.result = handShowing(0, 0.95);
    await run(pipeline, 4);
    mocks.result = handShowing(-0.8);
    await run(pipeline, 10, 2000); // later than the 200 ms cooldown
    expect(gestures).toEqual(['JUMP', 'LEFT']);
  });

  it('emits nothing for a resting, upright hand', async () => {
    const { pipeline, gestures } = await createPipeline();
    mocks.result = handShowing(0);
    await run(pipeline, 10);
    expect(gestures).toEqual([]);
    expect(pipeline.status.handPresent).toBe(true);
    expect(pipeline.status.stableGesture).toBeNull();
  });

  it('tracks how long the hand has been gone', async () => {
    const { pipeline } = await createPipeline();
    mocks.result = { kind: 'no-hand' };
    pipeline.step(1000);
    expect(pipeline.status.handPresent).toBe(false);
    expect(pipeline.handLostMs(1600)).toBe(600);
    mocks.result = handShowing(0);
    pipeline.step(1700);
    await flush();
    expect(pipeline.handLostMs(1800)).toBe(0);
  });

  it('ignores frames the camera has not refreshed', async () => {
    const { pipeline, gestures } = await createPipeline();
    mocks.result = { kind: 'stale' };
    await run(pipeline, 10);
    expect(gestures).toEqual([]);
    expect(mocks.detect).not.toHaveBeenCalled();
    expect(pipeline.status.cameraFps).toBe(0);
  });
});

describe('VisionPipeline resource handling', () => {
  it('releases only the camera on dispose; the shared models stay loaded', async () => {
    const { pipeline, loader } = await createPipeline();
    await pipeline.dispose();
    await pipeline.dispose();
    expect(mocks.stopCamera).toHaveBeenCalledTimes(1);
    expect(mocks.closeTracker).not.toHaveBeenCalled();

    // A second session reuses the already-loaded models instead of loading again.
    await createPipeline(loader);
    expect(mocks.createTracker).toHaveBeenCalledTimes(1);
  });

  it('reports a camera denial immediately, and keeps the models loading for a retry', async () => {
    const loader = makeLoader();
    mocks.openCamera.mockImplementationOnce(() => Promise.reject(new VisionError('camera-denied', 'no')));
    await expect(createPipeline(loader)).rejects.toMatchObject({ code: 'camera-denied' });

    const retry = await createPipeline(loader); // camera allowed this time
    expect(retry.pipeline.status.classifier).toBe('rules');
    expect(mocks.createTracker).toHaveBeenCalledTimes(1); // not loaded twice
  });

  it('stops the camera if the models fail to load', async () => {
    mocks.createTracker.mockImplementation(() => Promise.reject(new Error('model 404')));
    const loader = new ModelLoader({
      prefetch: () => Promise.resolve(),
      createTracker: mocks.createTracker as () => Promise<HandTracker>,
      loadClassifier: () => Promise.resolve(new RuleBasedClassifier()),
    });
    await expect(createPipeline(loader)).rejects.toMatchObject({ code: 'hand-model-failed' });
    expect(mocks.stopCamera).toHaveBeenCalledTimes(1);
  });

  it('reports progress that only moves forward, starting with the permission prompt', async () => {
    const { progress } = await createPipeline();
    expect(progress.length).toBeGreaterThan(1);
    expect(progress[0]?.message).toBe('Waiting for camera permission…');
    const fractions = progress.map((p) => p.fraction).filter((f): f is number => f !== null);
    expect(fractions).toEqual([...fractions].sort((a, b) => a - b));
    expect(fractions.at(-1)).toBe(1);
  });
});

describe('VisionPipeline adaptive load shedding', () => {
  it('analyses every 2nd frame when detection is slow, and recovers when it speeds up', async () => {
    const { pipeline } = await createPipeline();
    mocks.result = handShowing(0);

    // Slow detection: 20 ms per analysed frame.
    mocks.clock.detectCostMs = 20;
    let frame = 0;
    const step = async (count: number): Promise<void> => {
      for (let i = 0; i < count; i += 1) {
        pipeline.step(1000 + frame * 33);
        frame += 1;
        await flush();
      }
    };
    await step(ADAPTIVE_TRACKING.minSamples + 2);
    expect(pipeline.status.throttled).toBe(true);

    // While throttled, half the camera frames are skipped without analysis.
    mocks.detect.mockClear();
    mocks.skipFrame.mockClear();
    await step(20);
    expect(mocks.detect).toHaveBeenCalledTimes(10);
    expect(mocks.skipFrame).toHaveBeenCalledTimes(10);

    // Fast detection again: it should return to every frame.
    mocks.clock.detectCostMs = 2;
    await step(ADAPTIVE_TRACKING.minSamples * 4);
    expect(pipeline.status.throttled).toBe(false);
    mocks.detect.mockClear();
    await step(10);
    expect(mocks.detect).toHaveBeenCalledTimes(10);
  });

  it('does not throttle when detection is fast', async () => {
    const { pipeline } = await createPipeline();
    mocks.result = handShowing(0);
    mocks.clock.detectCostMs = 5;
    await run(pipeline, ADAPTIVE_TRACKING.minSamples * 3);
    expect(pipeline.status.throttled).toBe(false);
    expect(mocks.skipFrame).not.toHaveBeenCalled();
  });

  it('keeps response time close to normal while throttled: a switched gesture needs only 2 analysed frames', async () => {
    const { pipeline, gestures } = await createPipeline();
    mocks.result = handShowing(0.8);
    mocks.clock.detectCostMs = 20;
    await run(pipeline, ADAPTIVE_TRACKING.minSamples + 4);
    expect(pipeline.status.throttled).toBe(true);
    gestures.length = 0;

    mocks.result = handShowing(-0.8);
    // 4 camera frames = 2 analysed frames while throttled. With the normal 5-frame window
    // (3 votes needed) this would still be empty; with the throttled 3-frame window it is not.
    await run(pipeline, 4, 5000);
    expect(gestures).toEqual(['LEFT']);
  });

  it('still recognises a held gesture while throttled (shorter vote window)', async () => {
    const { pipeline, gestures } = await createPipeline();
    mocks.result = handShowing(0.8);
    mocks.clock.detectCostMs = 20;
    await run(pipeline, ADAPTIVE_TRACKING.minSamples + 4);
    expect(pipeline.status.throttled).toBe(true);
    gestures.length = 0;
    mocks.result = handShowing(-0.8);
    await run(pipeline, 12, 5000);
    expect(gestures).toEqual(['LEFT']);
  });
});
