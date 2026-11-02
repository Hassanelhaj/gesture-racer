import { createLogger } from '../logger';

const log = createLogger('storage/settings');
const STORAGE_KEY = 'gesture-racer:settings';

/** User-adjustable options. */
export interface Settings {
  /** Master sound on/off. */
  readonly sound: boolean;
  /** Show the (mirrored) camera preview bubble while driving. */
  readonly cameraPreview: boolean;
  /** Show latency / FPS / delegate readouts (also handy for the README metrics). */
  readonly diagnostics: boolean;
  /** Download and start the camera models in the background (about 22 MB). */
  readonly preload: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  sound: true,
  cameraPreview: true,
  diagnostics: false,
  preload: true,
};

/**
 * Loads settings from localStorage, tolerating a missing, corrupt or
 * partially-valid value. Each field falls back to its default independently,
 * so a future version adding a setting never discards the others.
 */
export function loadSettings(): Settings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) return DEFAULT_SETTINGS;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) return DEFAULT_SETTINGS;
    const record = parsed as Record<string, unknown>;
    const pick = (key: keyof Settings): boolean => {
      const value = record[key];
      return typeof value === 'boolean' ? value : DEFAULT_SETTINGS[key];
    };
    return {
      sound: pick('sound'),
      cameraPreview: pick('cameraPreview'),
      diagnostics: pick('diagnostics'),
      preload: pick('preload'),
    };
  } catch (error) {
    log.warn('Could not read settings; using defaults', error);
    return DEFAULT_SETTINGS;
  }
}

/** Persists settings; failure (storage blocked) is non-fatal. */
export function saveSettings(settings: Settings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch (error) {
    log.warn('Could not save settings', error);
  }
}
