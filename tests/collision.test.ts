import { describe, expect, it } from 'vitest';
import { collides, shrinkRect } from '../src/game/obstacles';
import type { Rect } from '../src/game/obstacles';

const box = (x: number, y: number, width = 100, height = 100): Rect => ({ x, y, width, height });

describe('shrinkRect', () => {
  it('insets every side by a fraction of its own size', () => {
    expect(shrinkRect(box(0, 0, 100, 200), 0.1)).toEqual({ x: 10, y: 20, width: 80, height: 160 });
  });

  it('is the identity at zero tolerance', () => {
    expect(shrinkRect(box(5, 6, 7, 8), 0)).toEqual(box(5, 6, 7, 8));
  });
});

describe('collides (tolerance hitbox)', () => {
  it('detects deep overlap', () => {
    expect(collides(box(0, 0), box(50, 50), 0.15)).toBe(true);
  });

  it('does not collide when apart', () => {
    expect(collides(box(0, 0), box(300, 0), 0.15)).toBe(false);
    expect(collides(box(0, 0), box(0, 300), 0.15)).toBe(false);
  });

  it('does not collide when merely touching edges', () => {
    expect(collides(box(0, 0), box(100, 0), 0)).toBe(false);
    expect(collides(box(0, 0), box(0, 100), 0)).toBe(false);
  });

  it('forgives a graze that plain AABB would count as a hit', () => {
    // 10 px of overlap on x: a hit with no tolerance, forgiven at 0.15 (15 px per side each).
    const a = box(0, 0);
    const b = box(90, 0);
    expect(collides(a, b, 0)).toBe(true);
    expect(collides(a, b, 0.15)).toBe(false);
  });

  it('still reports a hit once overlap exceeds the combined tolerance', () => {
    // Each box loses 15 px per side, so overlap must exceed 30 px.
    expect(collides(box(0, 0), box(69, 0), 0.15)).toBe(true);
    expect(collides(box(0, 0), box(70, 0), 0.15)).toBe(false);
    expect(collides(box(0, 0), box(71, 0), 0.15)).toBe(false);
  });

  it('applies the tolerance on the vertical axis too', () => {
    expect(collides(box(0, 0), box(0, 90), 0)).toBe(true);
    expect(collides(box(0, 0), box(0, 90), 0.15)).toBe(false);
  });

  it('is symmetric', () => {
    const a = box(10, 20, 56, 96);
    const b = box(40, 60, 56, 96);
    expect(collides(a, b)).toBe(collides(b, a));
  });

  it('defaults to the configured tolerance', () => {
    expect(collides(box(0, 0, 56, 96), box(0, 0, 56, 96))).toBe(true);
  });
});
