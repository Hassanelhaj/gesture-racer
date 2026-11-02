import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GestureModel } from '../src/vision/classifier';
import type { HandTracker } from '../src/vision/hands';
import { ModelLoader, prefetchAll, preloadVisionModels } from '../src/vision/loader';
import type { LoadProgress } from '../src/vision/loader';

function fakes() {
  const closeTracker = vi.fn();
  const tracker = { close: closeTracker } as unknown as HandTracker;
  const classifier = { kind: 'rules', busy: false, predict: vi.fn(), dispose: vi.fn(() => Promise.resolve()) } as unknown as GestureModel & {
    dispose: ReturnType<typeof vi.fn>;
  };
  return { tracker, classifier, closeTracker };
}

function makeLoader(overrides: Partial<ConstructorParameters<typeof ModelLoader>[0]> = {}) {
  const { tracker, classifier, closeTracker } = fakes();
  const deps = {
    prefetch: vi.fn((_urls: readonly string[], report: (loaded: number, total: number) => void) => {
      report(5, 20);
      report(20, 20);
      return Promise.resolve();
    }),
    createTracker: vi.fn(() => Promise.resolve(tracker)),
    loadClassifier: vi.fn(() => Promise.resolve(classifier as GestureModel)),
    ...overrides,
  };
  return { loader: new ModelLoader(deps), deps, tracker, classifier, closeTracker };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ModelLoader', () => {
  it('downloads first, then creates both models, and reports forward-only progress ending at 1', async () => {
    const { loader, deps } = makeLoader();
    const seen: LoadProgress[] = [];
    loader.subscribe((p) => seen.push(p));
    const models = await loader.load();
    expect(models.tracker).toBeDefined();
    expect(deps.prefetch).toHaveBeenCalledTimes(1);
    expect(deps.createTracker).toHaveBeenCalledTimes(1);
    const fractions = seen.map((p) => p.fraction);
    expect(fractions).toEqual([...fractions].sort((a, b) => a - b));
    expect(fractions.at(-1)).toBe(1);
    expect(seen.some((p) => p.message === 'Starting hand tracker…')).toBe(true);
  });

  it('shares one load between concurrent callers and later callers', async () => {
    const { loader, deps } = makeLoader();
    const [a, b] = await Promise.all([loader.load(), loader.load()]);
    expect(a).toBe(b);
    expect(await loader.load()).toBe(a);
    expect(deps.createTracker).toHaveBeenCalledTimes(1);
    expect(loader.started).toBe(true);
  });

  it('forgets a failed load so the next call retries from scratch', async () => {
    let attempts = 0;
    const { tracker } = fakes();
    const { loader, deps } = makeLoader({
      createTracker: vi.fn(() => {
        attempts += 1;
        return attempts === 1 ? Promise.reject(new Error('network')) : Promise.resolve(tracker);
      }),
    });
    await expect(loader.load()).rejects.toThrow('network');
    expect(loader.started).toBe(false);
    await expect(loader.load()).resolves.toBeDefined();
    expect(deps.prefetch).toHaveBeenCalledTimes(2);
  });

  it('does not leak the half that did load when the other fails', async () => {
    const first = makeLoader({ loadClassifier: vi.fn(() => Promise.reject(new Error('bad model'))) });
    await expect(first.loader.load()).rejects.toThrow('bad model');
    expect(first.closeTracker).toHaveBeenCalledTimes(1);

    const second = makeLoader({ createTracker: vi.fn(() => Promise.reject(new Error('no wasm'))) });
    await expect(second.loader.load()).rejects.toThrow('no wasm');
    expect(second.classifier.dispose).toHaveBeenCalledTimes(1);
  });

  it('release() closes both models and lets the next load start over', async () => {
    const { loader, deps, closeTracker, classifier } = makeLoader();
    await loader.load();
    await loader.release();
    expect(closeTracker).toHaveBeenCalledTimes(1);
    expect(classifier.dispose).toHaveBeenCalledTimes(1);
    expect(loader.started).toBe(false);
    await loader.load();
    expect(deps.createTracker).toHaveBeenCalledTimes(2);
  });

  it('release() before anything loaded is a no-op', async () => {
    const { loader } = makeLoader();
    await expect(loader.release()).resolves.toBeUndefined();
  });
});

describe('ModelLoader progress', () => {
  it('never moves backwards, even when a file turns out bigger than expected', async () => {
    const seen: number[] = [];
    const { loader } = makeLoader({
      prefetch: (_urls, report) => {
        report(10, 20); // 50 %
        report(10, 40); // a later Content-Length reveals more bytes: the ratio drops to 25 %
        report(40, 40);
        return Promise.resolve();
      },
    });
    loader.subscribe((progress) => seen.push(progress.fraction));
    await loader.load();
    for (let i = 1; i < seen.length; i += 1) expect(seen[i]).toBeGreaterThanOrEqual(seen[i - 1] ?? 0);
    expect(seen.at(-1)).toBe(1);
  });
});

describe('preloadVisionModels', () => {
  function stubIdle(): void {
    vi.stubGlobal('requestIdleCallback', (callback: () => void) => {
      callback();
      return 1;
    });
  }

  it('starts loading in the background when the browser is idle', () => {
    stubIdle();
    vi.stubGlobal('navigator', {});
    const { loader, deps } = makeLoader();
    preloadVisionModels(loader, 0);
    expect(deps.prefetch).toHaveBeenCalledTimes(1);
  });

  it('is skipped on data saver and slow connections', () => {
    stubIdle();
    for (const connection of [{ saveData: true }, { effectiveType: '2g' }, { effectiveType: '3g' }]) {
      vi.stubGlobal('navigator', { connection });
      const { loader, deps } = makeLoader();
      preloadVisionModels(loader, 0);
      expect(deps.prefetch).not.toHaveBeenCalled();
    }
  });

  it('does nothing if loading has already started', async () => {
    stubIdle();
    vi.stubGlobal('navigator', {});
    const { loader, deps } = makeLoader();
    await loader.load();
    preloadVisionModels(loader, 0);
    expect(deps.prefetch).toHaveBeenCalledTimes(1);
  });

  it('waits for the start delay, then loads once the browser is idle', () => {
    vi.useFakeTimers();
    try {
      stubIdle();
      vi.stubGlobal('navigator', {});
      const { loader, deps } = makeLoader();
      preloadVisionModels(loader, 1000);
      vi.advanceTimersByTime(999);
      expect(deps.prefetch).not.toHaveBeenCalled();
      vi.advanceTimersByTime(2);
      expect(deps.prefetch).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('does not start a second load if the player already started one during the delay', async () => {
    vi.useFakeTimers();
    try {
      stubIdle();
      vi.stubGlobal('navigator', {});
      const { loader, deps } = makeLoader();
      preloadVisionModels(loader, 1000);
      const loading = loader.load(); // the player clicked "Use my camera" early
      vi.advanceTimersByTime(1500);
      vi.useRealTimers();
      await loading;
      expect(deps.prefetch).toHaveBeenCalledTimes(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('swallows a background failure (it is retried when the player asks)', async () => {
    stubIdle();
    vi.stubGlobal('navigator', {});
    const { loader } = makeLoader({ createTracker: vi.fn(() => Promise.reject(new Error('offline'))) });
    expect(() => {
      preloadVisionModels(loader, 0);
    }).not.toThrow();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
});

describe('prefetchAll', () => {
  function bodyOf(chunks: number[]): ReadableStream<Uint8Array> {
    return new ReadableStream({
      start(controller) {
        for (const size of chunks) controller.enqueue(new Uint8Array(size));
        controller.close();
      },
    });
  }

  it('reports cumulative bytes against the advertised total', async () => {
    vi.stubGlobal('fetch', () => Promise.resolve(new Response(bodyOf([30, 70]), { headers: { 'content-length': '100' } })));
    const reports: [number, number][] = [];
    await prefetchAll(['/a.bin'], (loaded, total) => reports.push([loaded, total]));
    expect(reports.at(-1)).toEqual([100, 100]);
    expect(reports.map((r) => r[0])).toEqual([30, 100]);
  });

  it('never reports a total smaller than what has arrived (compressed responses)', async () => {
    vi.stubGlobal('fetch', () => Promise.resolve(new Response(bodyOf([500]), { headers: { 'content-length': '100', 'content-encoding': 'gzip' } })));
    const reports: [number, number][] = [];
    await prefetchAll(['/a.js'], (loaded, total) => reports.push([loaded, total]));
    for (const [loaded, total] of reports) expect(total).toBeGreaterThanOrEqual(loaded);
  });

  it('adds up several files', async () => {
    vi.stubGlobal('fetch', () => Promise.resolve(new Response(bodyOf([40]), { headers: { 'content-length': '40' } })));
    const reports: [number, number][] = [];
    await prefetchAll(['/a.bin', '/b.bin'], (loaded, total) => reports.push([loaded, total]));
    expect(reports.at(-1)?.[0]).toBe(80);
  });

  it('ignores failed and missing files instead of throwing', async () => {
    vi.stubGlobal('fetch', (url: string) => (url === '/gone.bin' ? Promise.resolve(new Response('no', { status: 404 })) : Promise.reject(new Error('offline'))));
    await expect(prefetchAll(['/gone.bin', '/down.bin'], () => undefined)).resolves.toBeUndefined();
  });
});
