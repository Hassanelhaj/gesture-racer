import { describe, expect, it } from 'vitest';
import { FEATURES } from '../src/constants';
import { FEATURE_NAMES, extractFeatures } from '../src/vision/features';
import { makeHand } from './helpers/makeHand';

const idx = (name: (typeof FEATURE_NAMES)[number]): number => FEATURE_NAMES.indexOf(name);

function features(spec: Parameters<typeof makeHand>[0] = {}, aspect = spec.aspect ?? 1): Float32Array {
  const vector = extractFeatures(makeHand(spec), aspect);
  if (vector === null) throw new Error('expected a feature vector');
  return vector;
}

describe('extractFeatures', () => {
  it('has one name per feature dimension', () => {
    expect(FEATURE_NAMES.length).toBe(FEATURES.dim);
  });

  it('returns a finite, bounded vector', () => {
    const v = features({ roll: 0.4, curl: 0.5 });
    expect(v.length).toBe(FEATURES.dim);
    for (const value of v) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThanOrEqual(-1);
      expect(value).toBeLessThanOrEqual(1);
    }
  });

  it('reads an upright open palm as roll 0 with straight fingers', () => {
    const v = features();
    expect(v[idx('roll_sin')]).toBeCloseTo(0, 5);
    expect(v[idx('roll_cos')]).toBeCloseTo(1, 5);
    for (const name of ['curl_index', 'curl_middle', 'curl_ring', 'curl_pinky'] as const) {
      expect(v[idx(name)]).toBeCloseTo(0, 3);
    }
  });

  it('encodes tilt to the right as positive roll_sin, left as negative (selfie view)', () => {
    const angle = Math.PI / 6;
    const right = features({ roll: angle });
    const left = features({ roll: -angle });
    expect(right[idx('roll_sin')]).toBeCloseTo(Math.sin(angle), 4);
    expect(left[idx('roll_sin')]).toBeCloseTo(-Math.sin(angle), 4);
    expect(right[idx('roll_cos')]).toBeCloseTo(Math.cos(angle), 4);
  });

  it('keeps (roll_sin, roll_cos) on the unit circle', () => {
    const v = features({ roll: 1.1 });
    const s = v[idx('roll_sin')] ?? 0;
    const c = v[idx('roll_cos')] ?? 0;
    expect(Math.hypot(s, c)).toBeCloseTo(1, 5);
  });

  it('recovers the requested curl from joint angles', () => {
    for (const curl of [0, 0.5, 1]) {
      const v = features({ curl });
      for (const name of ['curl_thumb', 'curl_index', 'curl_middle', 'curl_ring', 'curl_pinky'] as const) {
        expect(v[idx(name)]).toBeCloseTo(curl, 2);
      }
    }
  });

  it('separates an open hand from a fist by openness', () => {
    const open = features({ curl: 0 })[idx('hand_openness')] ?? 0;
    const fist = features({ curl: 1 })[idx('hand_openness')] ?? 0;
    expect(open).toBeGreaterThan(fist + 0.2);
  });

  it('is translation-invariant except for wrist position', () => {
    const a = features({ roll: 0.3, curl: 0.4, wrist: { x: 0.5, y: 0.5 } });
    const b = features({ roll: 0.3, curl: 0.4, wrist: { x: 0.9, y: 0.7 } });
    for (const [i, name] of FEATURE_NAMES.entries()) {
      if (name === 'wrist_x' || name === 'wrist_y') {
        expect(Math.abs((a[i] ?? 0) - (b[i] ?? 0))).toBeGreaterThan(0.05);
      } else {
        expect(b[i]).toBeCloseTo(a[i] ?? 0, 4);
      }
    }
  });

  it('is scale-invariant except for palm_scale (distance from camera)', () => {
    const near = features({ roll: 0.3, curl: 0.4, palmLength: 0.2 });
    const far = features({ roll: 0.3, curl: 0.4, palmLength: 0.1 });
    for (const [i, name] of FEATURE_NAMES.entries()) {
      if (name === 'palm_scale') {
        expect(near[i] ?? 0).toBeGreaterThan(far[i] ?? 0);
      } else if (name !== 'wrist_x' && name !== 'wrist_y') {
        expect(far[i]).toBeCloseTo(near[i] ?? 0, 4);
      }
    }
  });

  it('keeps curl, spread and openness unchanged when the hand rolls', () => {
    const upright = features({ curl: 0.6 });
    const tilted = features({ curl: 0.6, roll: 0.7 });
    for (const name of ['curl_thumb', 'curl_index', 'curl_pinky', 'finger_spread', 'hand_openness'] as const) {
      expect(tilted[idx(name)]).toBeCloseTo(upright[idx(name)] ?? 0, 4);
    }
  });

  it('corrects for aspect ratio so 16:9 and 4:3 cameras agree', () => {
    const wide = features({ roll: 0.5, curl: 0.3, aspect: 16 / 9 });
    const classic = features({ roll: 0.5, curl: 0.3, aspect: 4 / 3 });
    for (const [i, name] of FEATURE_NAMES.entries()) {
      if (name === 'wrist_x') continue; // normalised x legitimately differs per frame width
      expect(wide[i]).toBeCloseTo(classic[i] ?? 0, 4);
    }
  });

  it('clamps wrist position into 0..1 when the wrist is out of frame', () => {
    // Selfie-view x = -0.4 is off the left edge, y = 1.4 is below the bottom edge.
    const v = features({ wrist: { x: -0.4, y: 1.4 } });
    expect(v[idx('wrist_x')]).toBe(0);
    expect(v[idx('wrist_y')]).toBe(1);
  });

  it('returns null for degenerate geometry instead of NaNs', () => {
    const collapsed = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
    expect(extractFeatures(collapsed, 1)).toBeNull();
  });

  it('throws on programming errors', () => {
    expect(() => extractFeatures(makeHand().slice(0, 20), 1)).toThrow(RangeError);
    expect(() => extractFeatures(makeHand(), 0)).toThrow(RangeError);
    expect(() => extractFeatures(makeHand(), Number.NaN)).toThrow(RangeError);
  });
});
