import { CLASSIFIER, RULES } from '../constants';
import { GESTURES } from '../types';
import type { FeatureVector, Gesture, GesturePrediction } from '../types';
import type { GestureModel } from './classifier';
import { FEATURE_NAMES } from './features';

const ROLL_SIN = FEATURE_NAMES.indexOf('roll_sin');
const FINGER_CURLS = [
  FEATURE_NAMES.indexOf('curl_index'),
  FEATURE_NAMES.indexOf('curl_middle'),
  FEATURE_NAMES.indexOf('curl_ring'),
  FEATURE_NAMES.indexOf('curl_pinky'),
] as const;

const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));
const ramp = (value: number, start: number, full: number): number => clamp01((value - start) / (full - start));

/**
 * Classifies a feature vector with fixed geometric rules (see RULES in constants.ts).
 *
 *  - open hand (mean finger curl < openMaxCurl) tilted far enough -> LEFT or RIGHT
 *  - closed hand (mean finger curl high) -> JUMP
 *  - anything in between, or a hand held upright -> no gesture
 *
 * WHY it exists: the trained model needs the player's own recordings, so a
 * fresh checkout would otherwise show an error screen. These rules use the same
 * features the model uses, so the camera pipeline is identical either way.
 *
 * The returned `probabilities` are per-gesture *scores* in 0..1 that ramp
 * linearly with the evidence; they are not calibrated probabilities and do not
 * sum to 1. `confidence` is the best score, gated by `minConfidence` exactly
 * like the ONNX model's softmax output.
 */
export function scoreRules(
  features: FeatureVector,
  minConfidence: number = CLASSIFIER.minConfidence,
): Omit<GesturePrediction, 'inferenceMs'> {
  const rollSin = features[ROLL_SIN] ?? 0;
  let curlSum = 0;
  for (const index of FINGER_CURLS) curlSum += features[index] ?? 0;
  const meanCurl = curlSum / FINGER_CURLS.length;

  const isOpen = meanCurl < RULES.openMaxCurl;
  const tilt = isOpen ? ramp(Math.abs(rollSin), RULES.tiltStartSin, RULES.tiltFullSin) : 0;
  const scores: Record<Gesture, number> = {
    LEFT: rollSin < 0 ? tilt : 0,
    RIGHT: rollSin > 0 ? tilt : 0,
    JUMP: ramp(meanCurl, RULES.jumpStartCurl, RULES.jumpFullCurl),
  };

  let best: Gesture = GESTURES[0];
  for (const gesture of GESTURES) if (scores[gesture] > scores[best]) best = gesture;
  const confidence = scores[best];
  return {
    gesture: confidence >= minConfidence ? best : null,
    confidence,
    probabilities: scores,
  };
}

/** `GestureModel` wrapper around `scoreRules`; synchronous work behind the async interface. */
export class RuleBasedClassifier implements GestureModel {
  readonly kind = 'rules' as const;
  readonly busy = false;

  predict(features: FeatureVector): Promise<GesturePrediction> {
    const startedAt = performance.now();
    const result = scoreRules(features);
    return Promise.resolve({ ...result, inferenceMs: performance.now() - startedAt });
  }

  dispose(): Promise<void> {
    return Promise.resolve();
  }
}
