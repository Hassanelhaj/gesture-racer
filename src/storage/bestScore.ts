import { createLogger } from '../logger';

const log = createLogger('storage/bestScore');

const DB_NAME = 'gesture-racer';
const DB_VERSION = 1;
const STORE = 'kv';
const KEY = 'bestScore';

/**
 * In-memory copy. It is the source of truth when IndexedDB is unavailable
 * (private browsing in some browsers, blocked storage), so the game still
 * shows a best score for the current session instead of failing.
 */
let memoryBest = 0;
let dbPromise: Promise<IDBDatabase | null> | null = null;

/** Opens (once) the database, resolving null instead of throwing when unavailable. */
function openDb(): Promise<IDBDatabase | null> {
  dbPromise ??= new Promise<IDBDatabase | null>((resolve) => {
    // indexedDB is absent in some contexts; the DOM typings say it never is.
    const factory = (globalThis as { indexedDB?: IDBFactory }).indexedDB;
    if (factory === undefined) {
      resolve(null);
      return;
    }
    try {
      const request = factory.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        request.result.createObjectStore(STORE);
      };
      request.onsuccess = () => {
        resolve(request.result);
      };
      request.onerror = () => {
        log.warn('IndexedDB open failed; using in-memory best score', request.error);
        resolve(null);
      };
      request.onblocked = () => {
        resolve(null);
      };
    } catch (error) {
      log.warn('IndexedDB unavailable; using in-memory best score', error);
      resolve(null);
    }
  });
  return dbPromise;
}

const isValidScore = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;

/** Wraps an IDBRequest in a promise. */
function promisify<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => {
      resolve(request.result);
    };
    request.onerror = () => {
      reject(request.error ?? new Error('IndexedDB request failed'));
    };
  });
}

/** Reads the stored best score (0 if none or unreadable). */
export async function loadBestScore(): Promise<number> {
  const db = await openDb();
  if (db === null) return memoryBest;
  try {
    const stored: unknown = await promisify(db.transaction(STORE, 'readonly').objectStore(STORE).get(KEY));
    if (isValidScore(stored)) memoryBest = Math.max(memoryBest, Math.floor(stored));
  } catch (error) {
    log.warn('Could not read best score', error);
  }
  return memoryBest;
}

/**
 * Stores `score` if it beats the current best.
 * The compare-and-write happens inside ONE readwrite transaction so two tabs
 * finishing at once cannot overwrite a higher score with a lower one.
 *
 * @returns the best score after the call
 */
export async function saveBestScore(score: number): Promise<number> {
  const candidate = isValidScore(score) ? Math.floor(score) : 0;
  memoryBest = Math.max(memoryBest, candidate);
  const db = await openDb();
  if (db === null) return memoryBest;
  try {
    const store = db.transaction(STORE, 'readwrite').objectStore(STORE);
    const stored: unknown = await promisify(store.get(KEY));
    const current = isValidScore(stored) ? Math.floor(stored) : 0;
    const best = Math.max(current, candidate);
    if (best > current) await promisify(store.put(best, KEY));
    memoryBest = Math.max(memoryBest, best);
  } catch (error) {
    log.warn('Could not save best score', error);
  }
  return memoryBest;
}
