import { ASSETS, PRELOAD } from '../constants';
import { createLogger } from '../logger';
import type { GestureModel } from './classifier';
import { HandTracker } from './hands';
import { loadGestureModel } from './model';

const log = createLogger('vision/loader');

/** How far loading has got, for the progress bar. */
export interface LoadProgress {
  /** 0..1, never decreases during one load. */
  readonly fraction: number;
  readonly message: string;
}

/** The two heavy objects the camera pipeline needs. Shared across sessions. */
export interface VisionModels {
  readonly tracker: HandTracker;
  readonly classifier: GestureModel;
}

export interface LoaderDeps {
  /** Warms the HTTP cache for `urls`, reporting total bytes received / expected. */
  prefetch: (urls: readonly string[], report: (loaded: number, total: number) => void) => Promise<void>;
  createTracker: () => Promise<HandTracker>;
  loadClassifier: () => Promise<GestureModel>;
}

const ASSET_URLS: readonly string[] = [
  `${ASSETS.mediapipeWasmDir}/vision_wasm_internal.js`,
  `${ASSETS.mediapipeWasmDir}/vision_wasm_internal.wasm`,
  ASSETS.handModel,
];

function expectedBytes(url: string): number {
  if (url.endsWith('.wasm')) return PRELOAD.expectedBytes.wasm;
  if (url.endsWith('.js')) return PRELOAD.expectedBytes.js;
  if (url.endsWith('.task')) return PRELOAD.expectedBytes.model;
  return PRELOAD.expectedBytes.other;
}

/**
 * Streams every URL into the browser's HTTP cache while counting bytes.
 *
 * WHY: MediaPipe fetches these files itself, silently, so without this the
 * loading screen could only show a spinner for ~20 MB. Reading them first gives
 * a real progress bar, and lets the download start before the camera is even
 * requested; MediaPipe's own later request is then served from cache.
 * Failures are ignored on purpose: the real loader will report them properly.
 */
export async function prefetchAll(
  urls: readonly string[],
  report: (loaded: number, total: number) => void,
): Promise<void> {
  const loaded = new Map<string, number>();
  const totals = new Map<string, number>(urls.map((url) => [url, expectedBytes(url)]));
  const publish = (): void => {
    let sumLoaded = 0;
    let sumTotal = 0;
    for (const url of urls) {
      sumLoaded += loaded.get(url) ?? 0;
      sumTotal += Math.max(totals.get(url) ?? 0, loaded.get(url) ?? 0);
    }
    report(sumLoaded, sumTotal);
  };

  await Promise.all(
    urls.map(async (url) => {
      try {
        const response = await fetch(url);
        if (!response.ok || response.body === null) return;
        // Content-Length is only trustworthy for the decoded size when the body is not compressed.
        const length = Number(response.headers.get('content-length'));
        if (response.headers.get('content-encoding') === null && Number.isFinite(length) && length > 0) {
          totals.set(url, length);
        }
        const reader = response.body.getReader();
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          loaded.set(url, (loaded.get(url) ?? 0) + value.byteLength);
          publish();
        }
      } catch (error) {
        log.debug(`prefetch of ${url} failed (the real loader will report it)`, error);
      }
    }),
  );
}

const DEFAULT_DEPS: LoaderDeps = {
  prefetch: prefetchAll,
  createTracker: () => HandTracker.create(),
  loadClassifier: () => loadGestureModel(),
};

/**
 * Loads the hand tracker and the gesture classifier once and shares them.
 *
 * WHY a shared loader: creating them is the slow part (about 22 MB of download,
 * then WASM compilation). Doing it once, in the background, as soon as the page
 * is idle, means "Use my camera" usually finds everything ready; and returning
 * to the menu and starting again does not repeat the work. The *camera* is not
 * part of this: it is opened per session and switched off in the menu.
 */
export class ModelLoader {
  private readonly deps: LoaderDeps;
  private promise: Promise<VisionModels> | null = null;
  private current: LoadProgress = { fraction: 0, message: '' };
  private readonly listeners = new Set<(progress: LoadProgress) => void>();

  constructor(deps: LoaderDeps = DEFAULT_DEPS) {
    this.deps = deps;
  }

  /** Calls `listener` now and on every change. Returns an unsubscribe function. */
  subscribe(listener: (progress: LoadProgress) => void): () => void {
    this.listeners.add(listener);
    listener(this.current);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /** True once loading has started (or finished). */
  get started(): boolean {
    return this.promise !== null;
  }

  /**
   * Starts loading if needed and returns the shared result.
   * A failed load is forgotten, so the next call retries from scratch.
   */
  load(): Promise<VisionModels> {
    this.promise ??= this.run().catch((error: unknown) => {
      this.promise = null;
      this.publish({ fraction: 0, message: '' }, true);
      throw error;
    });
    return this.promise;
  }

  /** Closes the models, e.g. when leaving the page. The next `load()` starts over. */
  async release(): Promise<void> {
    const pending = this.promise;
    this.promise = null;
    if (pending === null) return;
    try {
      const { tracker, classifier } = await pending;
      tracker.close();
      await classifier.dispose();
    } catch {
      // Nothing was loaded, so there is nothing to release.
    }
  }

  private async run(): Promise<VisionModels> {
    this.publish({ fraction: 0, message: 'Downloading models…' }, true);
    await this.deps.prefetch(ASSET_URLS, (loaded, total) => {
      const share = total > 0 ? Math.min(1, loaded / total) : 0;
      this.publish({ fraction: PRELOAD.fetchWeight * share, message: 'Downloading models…' });
    });

    this.publish({ fraction: PRELOAD.fetchWeight, message: 'Starting hand tracker…' });
    const [tracker, classifier] = await Promise.allSettled([
      this.deps.createTracker(),
      this.deps.loadClassifier(),
    ]);
    if (tracker.status === 'rejected' || classifier.status === 'rejected') {
      // Do not leak whichever half did load.
      if (tracker.status === 'fulfilled') tracker.value.close();
      if (classifier.status === 'fulfilled') await classifier.value.dispose();
      throw tracker.status === 'rejected' ? tracker.reason : (classifier as PromiseRejectedResult).reason;
    }
    this.publish({ fraction: 1, message: 'Ready' });
    return { tracker: tracker.value, classifier: classifier.value };
  }

  private publish(next: LoadProgress, reset = false): void {
    // Progress only moves forward within one load, otherwise the bar would jitter.
    const fraction = reset ? next.fraction : Math.max(this.current.fraction, next.fraction);
    this.current = { fraction, message: next.message };
    for (const listener of this.listeners) listener(this.current);
  }
}

/** The app-wide loader. */
export const visionModels = new ModelLoader();

/** Whether background preloading is polite on this connection. */
function connectionAllowsPreload(): boolean {
  const connection = (navigator as { connection?: { saveData?: boolean; effectiveType?: string } }).connection;
  if (connection?.saveData === true) return false;
  return !['slow-2g', '2g', '3g'].includes(connection?.effectiveType ?? '');
}

/**
 * Starts loading the models in the background once the browser is idle, so they
 * are usually ready before the player clicks "Use my camera". Skipped on
 * data-saver and slow connections (those players load on demand instead).
 */
export function preloadVisionModels(
  loader: ModelLoader = visionModels,
  delayMs: number = PRELOAD.startDelayMs,
): void {
  if (loader.started || !connectionAllowsPreload()) return;
  const start = (): void => {
    if (loader.started) return; // the player clicked "Use my camera" during the delay
    loader.load().catch((error: unknown) => {
      log.debug('background preload failed; it will be retried on demand', error);
    });
  };
  const whenIdle = (): void => {
    if (typeof requestIdleCallback === 'function') requestIdleCallback(start, { timeout: PRELOAD.idleTimeoutMs });
    else setTimeout(start, PRELOAD.idleTimeoutMs);
  };
  if (delayMs > 0) setTimeout(whenIdle, delayMs);
  else whenIdle();
}
