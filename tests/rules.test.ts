import { describe, expect, it } from 'vitest';
import { CLASSIFIER } from '../src/constants';
import { extractFeatures } from '../src/vision/features';
import { RuleBasedClassifier, scoreRules } from '../src/vision/rules';
import { makeHand } from './helpers/makeHand';
import type { HandSpec } from './helpers/makeHand';

/** Features of a synthetic hand, viewed through a 4:3 camera. */
function classify(spec: HandSpec) {
  const features = extractFeatures(makeHand({ ...spec, aspect: 4 / 3 }), 4 / 3);
  if (features === null) throw new Error('expected features');
  return scoreRules(features);
}

const degrees = (d: number): number => (d * Math.PI) / 180;

describe('scoreRules', () => {
  it('reads a clearly tilted open hand as RIGHT / LEFT (selfie view)', () => {
    expect(classify({ roll: degrees(40), curl: 0.1 }).gesture).toBe('RIGHT');
    expect(classify({ roll: degrees(-40), curl: 0.1 }).gesture).toBe('LEFT');
  });

  it('reads a closed hand as JUMP, whatever its roll', () => {
    for (const roll of [0, degrees(30), degrees(-30)]) {
      expect(classify({ roll, curl: 0.95 }).gesture).toBe('JUMP');
    }
  });

  it('ignores a resting upright hand and small wobbles (dead zone)', () => {
    expect(classify({ roll: 0, curl: 0.1 }).gesture).toBeNull();
    expect(classify({ roll: degrees(8), curl: 0.1 }).gesture).toBeNull();
    expect(classify({ roll: degrees(-8), curl: 0.1 }).gesture).toBeNull();
  });

  it('starts registering a tilt at about 18 degrees (confidence 0.65 of the 9-24 degree ramp)', () => {
    expect(classify({ roll: degrees(16), curl: 0.1 }).gesture).toBeNull();
    expect(classify({ roll: degrees(20), curl: 0.1 }).gesture).toBe('RIGHT');
  });

  it('treats a half-closed hand as ambiguous, not as a gesture', () => {
    expect(classify({ roll: degrees(40), curl: 0.55 }).gesture).toBeNull();
    expect(classify({ roll: 0, curl: 0.5 }).gesture).toBeNull();
  });

  it('does not let a tilted fist steer', () => {
    const result = classify({ roll: degrees(45), curl: 0.95 });
    expect(result.gesture).toBe('JUMP');
    expect(result.probabilities.RIGHT).toBe(0);
  });

  it('is symmetric for left and right', () => {
    const right = classify({ roll: degrees(35), curl: 0.1 });
    const left = classify({ roll: degrees(-35), curl: 0.1 });
    expect(left.probabilities.LEFT).toBeCloseTo(right.probabilities.RIGHT, 5);
  });

  it('reports scores in 0..1 and a confidence equal to the best score', () => {
    const result = classify({ roll: degrees(35), curl: 0.1 });
    for (const score of Object.values(result.probabilities)) {
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
    expect(result.confidence).toBe(Math.max(...Object.values(result.probabilities)));
    expect(result.confidence).toBeGreaterThanOrEqual(CLASSIFIER.minConfidence);
  });

  it('works at different distances and positions (scale and translation do not matter)', () => {
    for (const palmLength of [0.1, 0.2, 0.3]) {
      expect(classify({ roll: degrees(40), curl: 0.1, palmLength, wrist: { x: 0.5, y: 0.9 } }).gesture).toBe('RIGHT');
    }
  });
});

describe('RuleBasedClassifier', () => {
  it('implements the model interface and matches scoreRules', async () => {
    const features = extractFeatures(makeHand({ roll: degrees(40), aspect: 4 / 3 }), 4 / 3);
    if (features === null) throw new Error('expected features');
    const model = new RuleBasedClassifier();
    expect(model.kind).toBe('rules');
    expect(model.busy).toBe(false);
    const prediction = await model.predict(features);
    expect(prediction.gesture).toBe(scoreRules(features).gesture);
    expect(prediction.inferenceMs).toBeGreaterThanOrEqual(0);
    await expect(model.dispose()).resolves.toBeUndefined();
  });
});
