import { describe, expect, it } from 'vitest';
import { CLASSIFIER } from '../src/constants';
import { GESTURES } from '../src/types';
import { decodeLogits, softmax } from '../src/vision/classifier';

describe('softmax', () => {
  it('returns probabilities that sum to 1', () => {
    const p = softmax([0.3, -1.2, 2.5]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(p.every((v) => v > 0 && v < 1)).toBe(true);
  });

  it('is numerically stable for huge logits (no NaN or Infinity)', () => {
    const p = softmax([1000, 999, -1000]);
    expect(p.every(Number.isFinite)).toBe(true);
    expect(p[0]).toBeGreaterThan(p[1] ?? 1);
    expect(p[2]).toBeCloseTo(0, 12);
  });

  it('is shift-invariant', () => {
    const a = softmax([1, 2, 3]);
    const b = softmax([101, 102, 103]);
    a.forEach((v, i) => {
      expect(b[i]).toBeCloseTo(v, 12);
    });
  });
});

describe('decodeLogits', () => {
  it('maps class indices to LEFT, RIGHT, JUMP in that order', () => {
    expect(GESTURES).toEqual(['LEFT', 'RIGHT', 'JUMP']);
    expect(decodeLogits([5, 0, 0]).gesture).toBe('LEFT');
    expect(decodeLogits([0, 5, 0]).gesture).toBe('RIGHT');
    expect(decodeLogits([0, 0, 5]).gesture).toBe('JUMP');
  });

  it('gates low-confidence predictions to null but still reports the confidence', () => {
    // Upright palm: LEFT and RIGHT nearly tied, so no class reaches the threshold.
    const decoded = decodeLogits([1.0, 0.95, -2]);
    expect(decoded.confidence).toBeLessThan(CLASSIFIER.minConfidence);
    expect(decoded.gesture).toBeNull();
    expect(decoded.confidence).toBeGreaterThan(0.4);
  });

  it('accepts a prediction exactly at the threshold', () => {
    const p = 0.7;
    const logit = Math.log(p / ((1 - p) / 2)); // class 0 gets exactly p, the rest share 1 - p
    expect(decodeLogits([logit, 0, 0], p - 1e-9).gesture).toBe('LEFT');
  });

  it('exposes a probability per gesture that sums to 1', () => {
    const { probabilities } = decodeLogits([2, 1, 0]);
    const total = GESTURES.reduce((sum, g) => sum + probabilities[g], 0);
    expect(total).toBeCloseTo(1, 12);
  });

  it('rejects a model with the wrong number of outputs', () => {
    expect(() => decodeLogits([1, 2])).toThrow(RangeError);
    expect(() => decodeLogits([1, 2, 3, 4])).toThrow(RangeError);
  });
});
