import { CAR, ROAD } from '../constants';
import type { Rect } from './obstacles';
import { MIDDLE_LANE, laneCenterX } from './road';

/**
 * The player's car: lane position (discrete target, smooth x) and a jump.
 *
 * WHY discrete lanes: gestures are coarse and delayed, so "move one lane" is a
 * far more forgiving control than continuous steering.
 */
export class Car {
  /** Target lane, 0 = leftmost. */
  lane: number = MIDDLE_LANE;
  /** Current centre x in world px; eases towards the target lane. */
  x: number = laneCenterX(MIDDLE_LANE);
  /** Seconds since the jump began, or null when on the ground. */
  private jumpElapsedS: number | null = null;

  reset(): void {
    this.lane = MIDDLE_LANE;
    this.x = laneCenterX(MIDDLE_LANE);
    this.jumpElapsedS = null;
  }

  /**
   * Moves one lane left (-1) or right (+1).
   * @returns false when already at the road edge (the request is ignored)
   */
  steer(direction: -1 | 1): boolean {
    const next = this.lane + direction;
    if (next < 0 || next >= ROAD.laneCount) return false;
    this.lane = next;
    return true;
  }

  /**
   * Starts a jump unless one is already in progress. Ignoring repeat requests
   * in mid-air stops a held fist from chaining jumps.
   * @returns whether a jump started
   */
  jump(): boolean {
    if (this.jumpElapsedS !== null) return false;
    this.jumpElapsedS = 0;
    return true;
  }

  update(dt: number): void {
    // Frame-rate independent exponential ease: same feel at 30 and 144 fps.
    const blend = 1 - Math.exp(-CAR.laneChangeRate * dt);
    this.x += (laneCenterX(this.lane) - this.x) * blend;

    if (this.jumpElapsedS !== null) {
      this.jumpElapsedS += dt;
      if (this.jumpElapsedS >= CAR.jumpDurationS) this.jumpElapsedS = null;
    }
  }

  /** Jump height 0..1 (sine arc: fast take-off, hang at the top, soft landing). */
  get height01(): number {
    if (this.jumpElapsedS === null) return 0;
    return Math.sin(Math.PI * (this.jumpElapsedS / CAR.jumpDurationS));
  }

  /** True while high enough to pass over a low obstacle such as a barrier. */
  get airborne(): boolean {
    return this.height01 >= CAR.airborneThreshold;
  }

  /** Full (un-shrunk) bounds in world px; collision applies the tolerance. */
  get rect(): Rect {
    return { x: this.x - CAR.width / 2, y: CAR.y, width: CAR.width, height: CAR.height };
  }
}
