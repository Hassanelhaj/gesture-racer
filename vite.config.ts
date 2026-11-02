import react from '@vitejs/plugin-react';
import type { Plugin } from 'vite';
import { defineConfig } from 'vitest/config';

/**
 * Content-Security-Policy for the production build only.
 *
 * WHY: "no video leaves the device" is the project's core promise. A CSP with
 * `connect-src 'self'` makes the browser itself enforce it: even a buggy or
 * compromised dependency cannot POST frames to another origin.
 *
 * ASSUMPTIONS (verify with `npm run build && npm run preview`):
 *  - 'wasm-unsafe-eval' is required to instantiate WebAssembly (MediaPipe, ORT).
 *  - blob: is allowed for script/worker because the WASM runtimes may spawn
 *    blob workers.
 *  - style-src needs 'unsafe-inline' because React sets inline style attributes.
 * It is NOT applied in dev: Vite's HMR client injects inline scripts that this
 * policy would block.
 */
const PRODUCTION_CSP = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval' blob:",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "media-src 'self' blob:",
  "connect-src 'self' blob: data:",
  "worker-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
].join('; ');

function productionCsp(): Plugin {
  return {
    name: 'gesture-racer:csp',
    apply: 'build',
    transformIndexHtml: (html) =>
      html.replace(
        '<!--CSP-->',
        `<meta http-equiv="Content-Security-Policy" content="${PRODUCTION_CSP}" />`,
      ),
  };
}

/**
 * Split the three heavy dependencies into their own chunks so the (tiny) app
 * code can be re-deployed without invalidating the browser cache of the big
 * runtimes.
 */
function manualChunks(id: string): string | undefined {
  if (!id.includes('/node_modules/')) return undefined;
  if (id.includes('onnxruntime-web')) return 'ort';
  if (id.includes('@mediapipe')) return 'mediapipe';
  if (id.includes('/react')) return 'react';
  return undefined;
}

export default defineConfig({
  // GitHub Pages serves project sites from /<repo>/. CI sets BASE_PATH; local
  // dev and `vite preview` use '/'.
  base: process.env.BASE_PATH ?? '/',
  plugins: [react(), productionCsp()],
  build: {
    target: 'es2022',
    sourcemap: false,
    // The ORT chunk alone exceeds the default 500 kB warning; that is expected.
    chunkSizeWarningLimit: 1024,
    rollupOptions: { output: { manualChunks } },
  },
  // ORT resolves its own .wasm/.mjs files at runtime; pre-bundling breaks that.
  optimizeDeps: { exclude: ['onnxruntime-web', 'onnxruntime-web/wasm'] },
  test: {
    include: ['tests/**/*.test.ts'],
    environment: 'node',
  },
});
