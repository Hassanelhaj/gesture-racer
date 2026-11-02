import { ROAD, SPEED, WORLD } from '../constants';

/** X of the road's left edge in world px (road is centred). */
export const ROAD_LEFT = (WORLD.width - ROAD.width) / 2;
export const ROAD_RIGHT = ROAD_LEFT + ROAD.width;
export const LANE_WIDTH = ROAD.width / ROAD.laneCount;
/** Distance after which the dashed lane markings repeat. */
export const DASH_PERIOD = ROAD.dashLength + ROAD.dashGap;
export const MIDDLE_LANE = Math.floor(ROAD.laneCount / 2);

/** Centre x of a lane (0 = leftmost). */
export function laneCenterX(lane: number): number {
  return ROAD_LEFT + LANE_WIDTH * (lane + 0.5);
}

/**
 * The scrolling road: owns the world speed, the distance driven and the
 * dash offset used for the lane markings.
 *
 * WHY the road owns speed: everything else (obstacles, score, audio pitch,
 * difficulty) derives from "how fast is the world moving", so there is exactly
 * one source of truth for it.
 */
export class Road {
  /** Current world scroll speed in px/s. */
  speed: number = SPEED.start;
  /** Total distance driven in px. */
  distance = 0;
  /** Lane-marking phase in 0..DASH_PERIOD, increases as the world scrolls down. */
  offset = 0;

  reset(): void {
    this.speed = SPEED.start;
    this.distance = 0;
    this.offset = 0;
  }

  /** 0 at start speed, 1 at max speed. Drives spawn density and audio pitch. */
  get difficulty(): number {
    return Math.min(1, Math.max(0, (this.speed - SPEED.start) / (SPEED.max - SPEED.start)));
  }

  /** Advances the world by `dt` seconds while accelerating towards max speed. */
  update(dt: number): void {
    this.speed = Math.min(SPEED.max, this.speed + SPEED.acceleration * dt);
    this.scroll(dt);
    this.distance += this.speed * dt;
  }

  /** Scrolls only the visuals at the current speed, e.g. behind the start screen. */
  scroll(dt: number): void {
    this.offset = (this.offset + this.speed * dt) % DASH_PERIOD;
  }
}
