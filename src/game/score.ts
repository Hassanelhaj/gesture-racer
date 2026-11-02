import { SCORE } from '../constants';

/**
 * Score = metres driven + a bonus per obstacle passed.
 *
 * WHY derived from distance: the score then rises faster as the car speeds up
 * without a separate "speed multiplier" rule, and it can never drift out of
 * sync with what the player sees.
 */
export class ScoreKeeper {
  private distancePx = 0;
  private bonus = 0;

  reset(): void {
    this.distancePx = 0;
    this.bonus = 0;
  }

  addDistance(px: number): void {
    this.distancePx += px;
  }

  addPassed(count: number): void {
    this.bonus += count * SCORE.passBonus;
  }

  /** Whole-number score. */
  get value(): number {
    return Math.floor(this.distancePx / SCORE.pxPerMeter) + this.bonus;
  }

  /** Distance driven in whole metres. */
  get meters(): number {
    return Math.floor(this.distancePx / SCORE.pxPerMeter);
  }
}

/** Converts world px/s to km/h for the HUD (25 px = 1 m). */
export function speedToKmh(pxPerSecond: number): number {
  return Math.round((pxPerSecond / SCORE.pxPerMeter) * 3.6);
}
