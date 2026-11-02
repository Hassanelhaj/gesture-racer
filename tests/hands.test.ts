import { afterEach, describe, expect, it, vi } from 'vitest';
import { FEATURES } from '../src/constants';
import { VISION_ERROR_MESSAGES, VisionError } from '../src/types';
import type { VisionErrorCode } from '../src/types';
import { openCamera, videoAspectRatio } from '../src/vision/hands';

/** A video element stand-in: only the members openCamera touches. */
function fakeVideo(play: () => Promise<void> = () => Promise.resolve()): HTMLVideoElement {
  return { play, srcObject: null, muted: false, playsInline: false } as unknown as HTMLVideoElement;
}

function stubGetUserMedia(behaviour: () => Promise<MediaStream>): void {
  vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: behaviour } });
}

async function codeOf(promise: Promise<unknown>): Promise<VisionErrorCode | 'resolved'> {
  try {
    await promise;
    return 'resolved';
  } catch (error) {
    if (error instanceof VisionError) return error.code;
    throw error;
  }
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('openCamera error mapping', () => {
  it.each([
    ['NotAllowedError', 'camera-denied'],
    ['SecurityError', 'camera-denied'],
    ['NotFoundError', 'camera-not-found'],
    ['OverconstrainedError', 'camera-not-found'],
    ['NotReadableError', 'camera-failed'],
    ['AbortError', 'camera-failed'],
  ] as const)('%s becomes %s', async (name, expected) => {
    stubGetUserMedia(() => Promise.reject(new DOMException('x', name)));
    expect(await codeOf(openCamera(fakeVideo()))).toBe(expected);
  });

  it('reports unsupported when mediaDevices is missing (insecure origin)', async () => {
    vi.stubGlobal('navigator', {});
    expect(await codeOf(openCamera(fakeVideo()))).toBe('camera-unsupported');
  });

  it('keeps the original browser error as the cause', async () => {
    const original = new DOMException('blocked', 'NotAllowedError');
    stubGetUserMedia(() => Promise.reject(original));
    await expect(openCamera(fakeVideo())).rejects.toMatchObject({ cause: original });
  });

  it('stops every track and clears the video if playback cannot start', async () => {
    const stop = vi.fn();
    const stream = { getTracks: () => [{ stop }], getVideoTracks: () => [] } as unknown as MediaStream;
    stubGetUserMedia(() => Promise.resolve(stream));
    const video = fakeVideo(() => Promise.reject(new Error('autoplay blocked')));
    expect(await codeOf(openCamera(video))).toBe('camera-failed');
    expect(stop).toHaveBeenCalledTimes(1);
    expect(video.srcObject).toBeNull();
  });

  it('releases the device on stop()', async () => {
    const stop = vi.fn();
    const track = { stop, addEventListener: vi.fn(), removeEventListener: vi.fn() };
    const stream = { getTracks: () => [track], getVideoTracks: () => [track] } as unknown as MediaStream;
    stubGetUserMedia(() => Promise.resolve(stream));
    const video = fakeVideo();
    const camera = await openCamera(video);
    expect(video.srcObject).toBe(stream);
    camera.stop();
    expect(stop).toHaveBeenCalledTimes(1);
    expect(video.srcObject).toBeNull();
  });
});

describe('videoAspectRatio', () => {
  it('uses the real video size', () => {
    expect(videoAspectRatio({ videoWidth: 1280, videoHeight: 720 } as HTMLVideoElement)).toBeCloseTo(16 / 9, 10);
  });

  it('falls back to the default before metadata has loaded', () => {
    expect(videoAspectRatio({ videoWidth: 0, videoHeight: 0 } as HTMLVideoElement)).toBe(FEATURES.defaultAspectRatio);
  });
});

describe('VISION_ERROR_MESSAGES', () => {
  it('has a non-empty player-facing message for every code', () => {
    for (const message of Object.values(VISION_ERROR_MESSAGES)) expect(message.length).toBeGreaterThan(20);
  });
});
