import { createLogger } from '../logger';
import { GestureClassifier } from './classifier';
import type { GestureModel } from './classifier';
import { RuleBasedClassifier } from './rules';

const log = createLogger('vision/model');

/**
 * Loads the trained ONNX classifier, or falls back to the built-in rules.
 *
 * WHY fall back instead of failing: a fresh checkout has no trained model yet
 * (it comes from the player's own recordings), and a deployed site could lose
 * the file. Either way the game stays playable. The fallback is NOT silent: the
 * returned model reports `kind: 'rules'`, which the practice screen and the
 * Diagnostics panel display.
 *
 * @param create injectable for tests; defaults to loading the ONNX model
 */
export async function loadGestureModel(
  create: () => Promise<GestureModel> = () => GestureClassifier.create(),
): Promise<GestureModel> {
  try {
    return await create();
  } catch (error) {
    log.warn('Trained gesture model unavailable; using the built-in rules instead', error);
    return new RuleBasedClassifier();
  }
}
