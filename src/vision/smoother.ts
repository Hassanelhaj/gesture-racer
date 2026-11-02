import { SMOOTHER } from '../constants';
import { GESTURES } from '../types';
import type { Gesture } from '../types';

/** What the smoother reports after each camera frame. */
export interface SmootherOutput {
  /** The stable gesture right now, or null when none is established. */
  readonly gesture: Gesture | null;
  /** True only on the frame where `gesture` differs from the previous frame's. */
  readonly changed: boolean;
}

export interface SmootherOptions {
  /** Number of most recent frames that vote. */
  readonly windowSize?: number;
  /** Minimum time between two emitted (non-null) gestures, in ms. */
  readonly cooldownMs?: number;
}

/**
 * Turns the classifier's noisy per-frame guesses into a stable gesture.
 *
 * WHY: a per-frame classifier flickers (one bad frame in thirty is enough to
 * veer the car into another lane). Two filters fix that:
 *
 *  1. MAJORITY VOTE over the last N frames. A gesture is "established" only
 *     when it has more than half the votes (3 of 5 by default). Frames with no
 *     gesture (null: low confidence, hand lost) are abstentions that still
 *     occupy a slot, so a gesture needs real agreement, not just a lack of
 *     rivals. If nothing has a majority the stable gesture is null.
 *
 *  2. COOLDOWN after each emitted gesture. A newly established gesture is held
 *     back until `cooldownMs` has passed since the previous one fired. This
 *     caps the action rate (200 ms = 5/s) and swallows LEFT<->RIGHT bounce
 *     while the hand passes through the middle. Falling back to null is never
 *     delayed, so releasing the gesture feels immediate.
 *
 * Time is passed in rather than read from a clock so behaviour is
 * deterministic in tests.
 */
export class GestureSmoother {
  private readonly windowSize: number;
  private readonly cooldownMs: number;
  private readonly majority: number;
  private readonly window: (Gesture | null)[] = [];
  private stable: Gesture | null = null;
  private lastEmitMs = Number.NEGATIVE_INFINITY;

  constructor(options: SmootherOptions = {}) {
    this.windowSize = options.windowSize ?? SMOOTHER.windowSize;
    this.cooldownMs = options.cooldownMs ?? SMOOTHER.cooldownMs;
    if (!Number.isInteger(this.windowSize) || this.windowSize < 1) {
      throw new RangeError(`windowSize must be a positive integer, got ${this.windowSize}`);
    }
    if (!Number.isFinite(this.cooldownMs) || this.cooldownMs < 0) {
      throw new RangeError(`cooldownMs must be >= 0, got ${this.cooldownMs}`);
    }
    this.majority = Math.floor(this.windowSize / 2) + 1;
  }

  /**
   * Feed one camera frame's observation.
   *
   * @param observation the classifier's gesture for this frame, or null for
   *        "no gesture / no hand". Call once per NEW camera frame, not per
   *        render frame, so the vote really spans N camera frames.
   * @param nowMs a monotonic timestamp in milliseconds
   */
  update(observation: Gesture | null, nowMs: number): SmootherOutput {
    this.window.push(observation);
    if (this.window.length > this.windowSize) this.window.shift();

    const candidate = this.majorityGesture();
    if (candidate === this.stable) return { gesture: this.stable, changed: false };

    if (candidate !== null && nowMs - this.lastEmitMs < this.cooldownMs) {
      // Established but still cooling down: keep the old value, retry next frame.
      return { gesture: this.stable, changed: false };
    }

    this.stable = candidate;
    if (candidate !== null) this.lastEmitMs = nowMs;
    return { gesture: this.stable, changed: true };
  }

  /**
   * Forget everything, e.g. after the hand has been gone long enough that the
   * old votes are meaningless. The cooldown is cleared too: a fresh hand
   * should respond immediately.
   */
  reset(): void {
    this.window.length = 0;
    this.stable = null;
    this.lastEmitMs = Number.NEGATIVE_INFINITY;
  }

  /** The gesture holding a strict majority of the window, or null. */
  private majorityGesture(): Gesture | null {
    for (const gesture of GESTURES) {
      let votes = 0;
      for (const entry of this.window) if (entry === gesture) votes += 1;
      if (votes >= this.majority) return gesture;
    }
    return null;
  }
}
