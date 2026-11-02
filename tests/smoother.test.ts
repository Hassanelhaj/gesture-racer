import { describe, expect, it } from 'vitest';
import { GestureSmoother } from '../src/vision/smoother';
import type { Gesture } from '../src/types';

const FRAME_MS = 33;

/** Feeds observations at ~30 fps from `startMs`; returns every output. */
function feed(
  smoother: GestureSmoother,
  observations: readonly (Gesture | null)[],
  startMs = 0,
): { gesture: Gesture | null; changed: boolean }[] {
  return observations.map((obs, i) => smoother.update(obs, startMs + i * FRAME_MS));
}

describe('GestureSmoother majority vote', () => {
  it('needs 3 of 5 frames to establish a gesture', () => {
    const s = new GestureSmoother();
    const out = feed(s, ['LEFT', 'LEFT', 'LEFT']);
    expect(out[0]?.gesture).toBeNull();
    expect(out[1]?.gesture).toBeNull();
    expect(out[2]).toEqual({ gesture: 'LEFT', changed: true });
  });

  it('reports `changed` only on the transition frame', () => {
    const s = new GestureSmoother();
    const out = feed(s, ['LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT']);
    expect(out.map((o) => o.changed)).toEqual([false, false, true, false, false]);
  });

  it('ignores a single flickering frame', () => {
    const s = new GestureSmoother();
    const out = feed(s, ['LEFT', 'LEFT', 'LEFT', 'RIGHT', 'LEFT', 'LEFT']);
    expect(out.slice(2).every((o) => o.gesture === 'LEFT')).toBe(true);
    expect(out.filter((o) => o.changed).length).toBe(1);
  });

  it('treats null frames as abstentions that still occupy window slots', () => {
    const s = new GestureSmoother();
    const out = feed(s, ['LEFT', null, 'LEFT', null, null]);
    expect(out.every((o) => o.gesture === null)).toBe(true);
  });

  it('yields null when no gesture holds a majority', () => {
    const s = new GestureSmoother();
    const out = feed(s, ['LEFT', 'RIGHT', 'JUMP', 'LEFT', 'RIGHT']);
    expect(out.at(-1)?.gesture).toBeNull();
  });

  it('releases to null without delay once the majority is lost', () => {
    const s = new GestureSmoother();
    const out = feed(s, ['LEFT', 'LEFT', 'LEFT', null, null, null]);
    expect(out[3]?.gesture).toBe('LEFT'); // window still L,L,L,null
    expect(out[4]?.gesture).toBe('LEFT'); // L,L,L,null,null -> 3 votes
    expect(out[5]).toEqual({ gesture: null, changed: true }); // L,L,null,null,null
  });

  it('honours a custom window size', () => {
    const s = new GestureSmoother({ windowSize: 3, cooldownMs: 0 });
    const out = feed(s, ['JUMP', 'JUMP']);
    expect(out[1]).toEqual({ gesture: 'JUMP', changed: true });
  });
});

describe('GestureSmoother cooldown', () => {
  it('holds back a new gesture until the cooldown has elapsed', () => {
    const s = new GestureSmoother({ windowSize: 3, cooldownMs: 200 });
    s.update('LEFT', 0);
    expect(s.update('LEFT', 33)).toEqual({ gesture: 'LEFT', changed: true }); // fires at t=33
    // RIGHT wins the vote almost immediately, but 200 ms have not passed.
    s.update('RIGHT', 66);
    expect(s.update('RIGHT', 100)).toEqual({ gesture: 'LEFT', changed: false });
    expect(s.update('RIGHT', 200)).toEqual({ gesture: 'LEFT', changed: false });
    // t = 33 + 200 = 233: allowed.
    expect(s.update('RIGHT', 233)).toEqual({ gesture: 'RIGHT', changed: true });
  });

  it('applies the cooldown across a release to null (rate limit, not debounce)', () => {
    const s = new GestureSmoother({ windowSize: 3, cooldownMs: 200 });
    s.update('JUMP', 0);
    s.update('JUMP', 10); // fires at t=10
    s.update(null, 20);
    expect(s.update(null, 30)).toEqual({ gesture: null, changed: true });
    s.update('JUMP', 40);
    expect(s.update('JUMP', 50)).toEqual({ gesture: null, changed: false }); // blocked
    expect(s.update('JUMP', 210)).toEqual({ gesture: 'JUMP', changed: true });
  });

  it('does not delay the very first gesture', () => {
    const s = new GestureSmoother();
    const out = feed(s, ['RIGHT', 'RIGHT', 'RIGHT'], 5);
    expect(out[2]).toEqual({ gesture: 'RIGHT', changed: true });
  });

  it('reset() clears votes, state and cooldown', () => {
    const s = new GestureSmoother({ windowSize: 3, cooldownMs: 1000 });
    s.update('LEFT', 0);
    s.update('LEFT', 10);
    s.reset();
    s.update('RIGHT', 20);
    expect(s.update('RIGHT', 30)).toEqual({ gesture: 'RIGHT', changed: true });
  });
});

describe('GestureSmoother options', () => {
  it('rejects invalid options', () => {
    expect(() => new GestureSmoother({ windowSize: 0 })).toThrow(RangeError);
    expect(() => new GestureSmoother({ windowSize: 2.5 })).toThrow(RangeError);
    expect(() => new GestureSmoother({ cooldownMs: -1 })).toThrow(RangeError);
  });
});
