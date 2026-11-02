/**
 * Prepares the self-hosted runtime assets under public/.
 *
 * WHY self-host: (1) the app works offline after first load, (2) the CSP can
 * stay at `connect-src 'self'`, and (3) the privacy claim does not depend on a
 * third-party CDN staying honest.
 *
 * Idempotent: safe to run before every dev/build; it skips work already done.
 *
 *  - MediaPipe WASM  : copied from node_modules -> public/wasm/mediapipe
 *  - hand_landmarker : downloaded once          -> public/models/hand_landmarker.task
 *
 * ONNX Runtime's WASM is NOT handled here. ORT loads its loader script with a
 * dynamic import(), which Vite's dev server refuses for files in public/, so
 * vision/classifier.ts imports both files with `?url` and Vite bundles them.
 */
import { copyFile, mkdir, rename, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const PUBLIC_DIR = path.join(ROOT, 'public');

// Assumption: Google's float16 v1 hand model is the one MediaPipe Tasks documents.
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
// Sanity floor: the real file is roughly 7.5 MB; anything tiny is an error page.
const HAND_MODEL_MIN_BYTES = 1_000_000;

// MediaPipe picks "vision_wasm_internal" when the browser supports WASM SIMD (every
// browser since 2021). Its "_module" and "_nosimd" variants are ~13 MB each and would
// never be requested by this app, so they are not shipped. A pre-SIMD browser gets
// the normal "hand model failed to load" error screen instead of a half-working game.
const MEDIAPIPE_FILES = ['vision_wasm_internal.js', 'vision_wasm_internal.wasm'];

async function exists(file) {
  try {
    await stat(file);
    return true;
  } catch {
    return false;
  }
}

async function copyMediapipeWasm() {
  const src = path.join(ROOT, 'node_modules/@mediapipe/tasks-vision/wasm');
  const dest = path.join(PUBLIC_DIR, 'wasm/mediapipe');
  if (!(await exists(src))) throw new Error(`Missing ${src}. Run "npm install" first.`);
  await mkdir(dest, { recursive: true });
  for (const name of MEDIAPIPE_FILES) {
    const from = path.join(src, name);
    if (!(await exists(from))) {
      throw new Error(`${from} not found. The @mediapipe/tasks-vision file layout may have changed.`);
    }
    await copyFile(from, path.join(dest, name));
  }
  console.log(`[assets] MediaPipe WASM ready (${MEDIAPIPE_FILES.length} files)`);
}

async function fetchHandModel() {
  const dest = path.join(PUBLIC_DIR, 'models/hand_landmarker.task');
  if (await exists(dest)) {
    console.log('[assets] hand_landmarker.task already present');
    return;
  }
  await mkdir(path.dirname(dest), { recursive: true });
  console.log('[assets] downloading hand_landmarker.task ...');
  const response = await fetch(HAND_MODEL_URL);
  if (!response.ok) throw new Error(`Download failed: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length < HAND_MODEL_MIN_BYTES) {
    throw new Error(`Downloaded file is only ${bytes.length} bytes; expected a ~7.5 MB model.`);
  }
  // Write to a temp name first so an interrupted download is never mistaken for a good file.
  const tmp = `${dest}.part`;
  await writeFile(tmp, bytes);
  await rename(tmp, dest);
  console.log(`[assets] hand_landmarker.task saved (${(bytes.length / 1e6).toFixed(1)} MB)`);
}

try {
  await copyMediapipeWasm();
  await fetchHandModel();
} catch (error) {
  console.error(`[assets] ${error instanceof Error ? error.message : String(error)}`);
  console.error(`[assets] Manual fallback: download ${HAND_MODEL_URL}`);
  console.error('[assets] and save it as public/models/hand_landmarker.task');
  process.exit(1);
}
