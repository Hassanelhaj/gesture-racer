import type * as OrtModule from 'onnxruntime-web/wasm';
import ortLoaderUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import { ASSETS, CLASSIFIER, FEATURES } from '../constants';
import { createLogger } from '../logger';
import { GESTURES, VisionError } from '../types';
import type { FeatureVector, Gesture, GesturePrediction } from '../types';

const log = createLogger('vision/classifier');

/**
 * Numerically stable softmax. Subtracting the max logit first prevents exp()
 * overflow; the model exports raw logits so INT8 quantisation never has to
 * touch a softmax node.
 */
export function softmax(logits: ArrayLike<number>): number[] {
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < logits.length; i += 1) max = Math.max(max, logits[i] ?? max);
  const exps: number[] = [];
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const e = Math.exp((logits[i] ?? max) - max);
    exps.push(e);
    sum += e;
  }
  return exps.map((e) => e / sum);
}

/**
 * Converts raw model logits into a prediction, applying the confidence gate.
 *
 * WHY the gate: the model has no NEUTRAL class, so it must answer LEFT, RIGHT
 * or JUMP even for a hand that is just resting. A resting upright palm sits
 * between LEFT and RIGHT and yields split probabilities, which the gate turns
 * into "no gesture".
 *
 * @throws RangeError if the logits length differs from the number of gestures
 */
export function decodeLogits(
  logits: ArrayLike<number>,
  minConfidence: number = CLASSIFIER.minConfidence,
): Omit<GesturePrediction, 'inferenceMs'> {
  if (logits.length !== GESTURES.length) {
    throw new RangeError(`Model returned ${logits.length} logits, expected ${GESTURES.length}`);
  }
  const probs = softmax(logits);
  let bestIndex = 0;
  for (let i = 1; i < probs.length; i += 1) {
    if ((probs[i] ?? 0) > (probs[bestIndex] ?? 0)) bestIndex = i;
  }
  const confidence = probs[bestIndex] ?? 0;
  const probabilities = {} as Record<Gesture, number>;
  GESTURES.forEach((gesture, i) => {
    probabilities[gesture] = probs[i] ?? 0;
  });
  const best = GESTURES[bestIndex] ?? null;
  return {
    gesture: confidence >= minConfidence ? best : null,
    confidence,
    probabilities,
  };
}

/**
 * What the pipeline needs from any gesture classifier. Two implementations:
 * the trained ONNX model (`GestureClassifier`) and the built-in rules
 * (`RuleBasedClassifier`, used when no trained model is installed).
 */
export interface GestureModel {
  readonly kind: 'onnx' | 'rules';
  /** True while an inference is running; the caller drops frames instead of queueing. */
  readonly busy: boolean;
  predict: (features: FeatureVector) => Promise<GesturePrediction>;
  dispose: () => Promise<void>;
}

/** The ONNX Runtime module, which is loaded lazily (see `loadOrt`). */
type Ort = typeof OrtModule;

let ortPromise: Promise<Ort> | null = null;

/**
 * Loads ONNX Runtime Web on first use and configures it once.
 *
 * WHY lazy: ORT's WASM is ~15 MB. Players without a trained model never need it
 * (the built-in rules run instead), and a dynamic import keeps both the
 * JavaScript and the WASM download out of their first load entirely.
 *
 *  - wasmPaths: point ORT at the loader/.wasm that Vite bundles as hashed assets
 *    (`?url` imports). They are served from our own origin, never a CDN (see CSP
 *    in vite.config.ts). Using `?url` instead of files in public/ matters
 *    because Vite's dev server refuses to `import()` a script from public/,
 *    which is exactly how ORT loads its loader.
 *  - numThreads = 1: multi-threaded WASM needs cross-origin isolation, which
 *    GitHub Pages cannot provide. A ~1k-parameter MLP gains nothing from threads.
 *  - proxy = false: no worker round-trip; keeps latency minimal.
 */
function loadOrt(): Promise<Ort> {
  ortPromise ??= import('onnxruntime-web/wasm').then((ort) => {
    ort.env.wasm.wasmPaths = { mjs: ortLoaderUrl, wasm: ortWasmUrl };
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    return ort;
  });
  // A failed import (offline) must not be cached forever.
  ortPromise.catch(() => {
    ortPromise = null;
  });
  return ortPromise;
}

/**
 * Downloads the model file BEFORE touching ONNX Runtime.
 *
 * WHY: if the file is missing we want to know in milliseconds and fall back to
 * the rules, instead of first downloading and compiling 15 MB of WASM only to
 * discover there is nothing to run. Also rejects an HTML page served with a 200
 * (static hosts and dev servers often answer unknown paths with index.html).
 */
async function fetchModelBytes(modelUrl: string): Promise<Uint8Array> {
  const response = await fetch(modelUrl);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  if ((response.headers.get('content-type') ?? '').includes('text/html')) {
    throw new Error('Got an HTML page instead of a model file');
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.length === 0) throw new Error('Model file is empty');
  return bytes;
}

/**
 * ONNX gesture classifier running on ONNX Runtime Web's WASM backend.
 *
 * WASM-only is deliberate: for a model this small, WebGPU's dispatch and
 * read-back overhead exceeds the compute, and it would add a 2nd runtime
 * variant and a failure mode (no WebGPU) for no gain. See docs/decisions.md.
 */
export class GestureClassifier implements GestureModel {
  readonly kind = 'onnx' as const;
  private readonly ort: Ort;
  private readonly session: OrtModule.InferenceSession;
  private readonly inputName: string;
  private readonly outputName: string;
  private inFlight = false;

  private constructor(ort: Ort, session: OrtModule.InferenceSession, inputName: string, outputName: string) {
    this.ort = ort;
    this.session = session;
    this.inputName = inputName;
    this.outputName = outputName;
  }

  /**
   * Loads the model and runs one warm-up inference.
   *
   * ORDER MATTERS FOR LOAD TIME: the (tiny) model file is fetched and checked
   * FIRST. Only if it is really there is the 15 MB ONNX Runtime WebAssembly
   * downloaded and started. When no trained model is installed this fails in
   * milliseconds and the app uses the built-in rules without ever touching
   * ONNX Runtime. (Previously it downloaded and initialised the runtime and
   * only then discovered there was nothing to run.)
   *
   * The warm-up inference (a) fails fast if the model's input/output shape does
   * not match the app's contract and (b) pays WASM first-run cost during
   * loading instead of on the player's first gesture.
   *
   * @throws VisionError('classifier-load-failed')
   */
  static async create(modelUrl: string = ASSETS.gestureModel): Promise<GestureClassifier> {
    try {
      const bytes = await fetchModelBytes(modelUrl);
      const ort = await loadOrt();
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      const inputName = session.inputNames[0];
      const outputName = session.outputNames[0];
      if (inputName === undefined || outputName === undefined) {
        throw new Error('Model must have one input and one output');
      }
      const classifier = new GestureClassifier(ort, session, inputName, outputName);
      await classifier.predict(new Float32Array(FEATURES.dim));
      log.info('classifier ready', { input: inputName, output: outputName });
      return classifier;
    } catch (cause) {
      log.warn(
        `Could not load the gesture model from ${modelUrl}. Running locally? ` +
          'Train and export it first (see ml/README.md); until then the built-in rules are used.',
        cause,
      );
      throw new VisionError('classifier-load-failed', `Could not load gesture model from ${modelUrl}`, { cause });
    }
  }

  /**
   * True while an inference is running. ORT sessions are not meant for
   * overlapping runs, so the game loop should skip a frame rather than queue
   * (a late gesture is worse than a dropped one).
   */
  get busy(): boolean {
    return this.inFlight;
  }

  /**
   * Classifies one feature vector.
   *
   * @throws RangeError if the vector length is wrong (programming error)
   * @throws VisionError('classifier-inference-failed') on runtime failure
   */
  async predict(features: FeatureVector): Promise<GesturePrediction> {
    if (features.length !== FEATURES.dim) {
      throw new RangeError(`Expected ${FEATURES.dim} features, got ${features.length}`);
    }
    this.inFlight = true;
    try {
      const startedAt = performance.now();
      // Copy: ORT may hold the buffer until the run resolves.
      const input = new this.ort.Tensor('float32', Float32Array.from(features), [1, FEATURES.dim]);
      const outputs = await this.session.run({ [this.inputName]: input });
      const inferenceMs = performance.now() - startedAt;

      const output = outputs[this.outputName];
      if (output === undefined || !(output.data instanceof Float32Array)) {
        throw new Error('Model output is missing or not float32');
      }
      if (inferenceMs > CLASSIFIER.inferenceBudgetMs) {
        log.debug(`inference over budget: ${inferenceMs.toFixed(1)} ms`);
      }
      return { ...decodeLogits(output.data), inferenceMs };
    } catch (cause) {
      throw new VisionError('classifier-inference-failed', 'Gesture inference failed', { cause });
    } finally {
      this.inFlight = false;
    }
  }

  /** Releases the WASM session memory. */
  async dispose(): Promise<void> {
    await this.session.release();
  }
}
