import 'fake-indexeddb/auto';
import { IDBFactory } from 'fake-indexeddb';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type * as BestScoreModule from '../src/storage/bestScore';
import { DEFAULT_SETTINGS } from '../src/storage/settings';

/** Fresh module state per test: bestScore keeps a module-level cache and DB handle. */
async function freshBestScore(): Promise<typeof BestScoreModule> {
  vi.resetModules();
  return import('../src/storage/bestScore');
}

describe('bestScore with IndexedDB', () => {
  beforeEach(() => {
    vi.stubGlobal('indexedDB', new IDBFactory());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('starts at 0', async () => {
    const { loadBestScore } = await freshBestScore();
    expect(await loadBestScore()).toBe(0);
  });

  it('keeps only the highest score', async () => {
    const { loadBestScore, saveBestScore } = await freshBestScore();
    expect(await saveBestScore(120)).toBe(120);
    expect(await saveBestScore(80)).toBe(120);
    expect(await saveBestScore(121)).toBe(121);
    expect(await loadBestScore()).toBe(121);
  });

  it('persists across a reload (a new module instance on the same database)', async () => {
    const first = await freshBestScore();
    await first.saveBestScore(300);
    const second = await freshBestScore();
    expect(await second.loadBestScore()).toBe(300);
  });

  it('never overwrites a higher stored score with a lower one (checked after a reload)', async () => {
    const first = await freshBestScore();
    await first.saveBestScore(120);
    await first.saveBestScore(80);
    const second = await freshBestScore(); // new module = empty memory cache, so only the database speaks
    expect(await second.loadBestScore()).toBe(120);
  });

  it('floors fractional scores and ignores junk input', async () => {
    const { saveBestScore } = await freshBestScore();
    expect(await saveBestScore(42.9)).toBe(42);
    expect(await saveBestScore(Number.NaN)).toBe(42);
    expect(await saveBestScore(-5)).toBe(42);
  });
});

describe('bestScore without IndexedDB (private mode / blocked storage)', () => {
  beforeEach(() => {
    vi.stubGlobal('indexedDB', undefined);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('still tracks the best score for the session in memory', async () => {
    const { loadBestScore, saveBestScore } = await freshBestScore();
    expect(await loadBestScore()).toBe(0);
    expect(await saveBestScore(55)).toBe(55);
    expect(await saveBestScore(10)).toBe(55);
    expect(await loadBestScore()).toBe(55);
  });
});

/** A minimal localStorage backed by a Map. */
function stubLocalStorage(initial: Record<string, string> = {}, failWrites = false): Map<string, string> {
  const store = new Map(Object.entries(initial));
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => {
      if (failWrites) throw new DOMException('quota', 'QuotaExceededError');
      store.set(key, value);
    },
  });
  return store;
}

describe('settings', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns the defaults when nothing is stored', async () => {
    stubLocalStorage();
    const { loadSettings } = await import('../src/storage/settings');
    expect(loadSettings()).toEqual(DEFAULT_SETTINGS);
  });

  it('round-trips a saved value', async () => {
    stubLocalStorage();
    const { loadSettings, saveSettings } = await import('../src/storage/settings');
    saveSettings({ sound: false, cameraPreview: false, diagnostics: true, preload: false });
    expect(loadSettings()).toEqual({ sound: false, cameraPreview: false, diagnostics: true, preload: false });
  });

  it('turns preloading on by default, including for settings saved before it existed', async () => {
    stubLocalStorage({ 'gesture-racer:settings': JSON.stringify({ sound: false, cameraPreview: true, diagnostics: false }) });
    const { loadSettings } = await import('../src/storage/settings');
    expect(loadSettings().preload).toBe(true);
  });

  it('falls back per field when the stored value is partly invalid', async () => {
    stubLocalStorage({ 'gesture-racer:settings': JSON.stringify({ sound: false, cameraPreview: 'yes' }) });
    const { loadSettings } = await import('../src/storage/settings');
    expect(loadSettings()).toEqual({ ...DEFAULT_SETTINGS, sound: false });
  });

  it.each(['not json', 'null', '42', '"text"'])('survives corrupt storage (%s)', async (raw) => {
    stubLocalStorage({ 'gesture-racer:settings': raw });
    const { loadSettings } = await import('../src/storage/settings');
    expect(loadSettings()).toEqual(DEFAULT_SETTINGS);
  });

  it('does not throw when storage is blocked or full', async () => {
    stubLocalStorage({}, true);
    const { saveSettings } = await import('../src/storage/settings');
    expect(() => {
      saveSettings(DEFAULT_SETTINGS);
    }).not.toThrow();
  });

  it('does not throw when localStorage itself is unavailable', async () => {
    vi.stubGlobal('localStorage', undefined);
    const { loadSettings } = await import('../src/storage/settings');
    expect(loadSettings()).toEqual(DEFAULT_SETTINGS);
  });
});
