import { CAR, COLLISION, OBSTACLES, ROAD, WORLD } from '../constants';
import type { Car } from './car';
import { MIDDLE_LANE, laneCenterX } from './road';
import type { Road } from './road';

/** Axis-aligned rectangle in world px (x, y = top-left). */
export interface Rect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

/** Returns `rect` inset by `tolerance` (a fraction of its size) on every side. */
export function shrinkRect(rect: Rect, tolerance: number): Rect {
  const dx = rect.width * tolerance;
  const dy = rect.height * tolerance;
  return { x: rect.x + dx, y: rect.y + dy, width: rect.width - 2 * dx, height: rect.height - 2 * dy };
}

/**
 * Overlap test with forgiving hitboxes.
 *
 * WHY shrink both boxes: sprites are drawn with rounded corners, mirrors and
 * glow, so their bounding boxes are larger than what the player perceives as
 * "the car". Testing the inner core stops unfair deaths on graze contact.
 * Touching edges do NOT collide.
 *
 * @param tolerance per-side inset as a fraction of each rect's size (0..0.5)
 */
export function collides(
  a: Rect,
  b: Rect,
  tolerance: number = COLLISION.hitboxTolerance,
): boolean {
  const ra = shrinkRect(a, tolerance);
  const rb = shrinkRect(b, tolerance);
  return (
    ra.x < rb.x + rb.width &&
    ra.x + ra.width > rb.x &&
    ra.y < rb.y + rb.height &&
    ra.y + ra.height > rb.y
  );
}

/** A function returning a float in [0, 1), like Math.random. */
export type Rng = () => number;

/**
 * Small seedable PRNG (mulberry32). Used so tests and replays are
 * deterministic; the real game seeds it from the clock.
 */
export function createRng(seed: number): Rng {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A car blocks its lane; a barrier is low enough to jump over. */
export type ObstacleKind = 'car' | 'barrier';

export interface Obstacle {
  readonly id: number;
  readonly kind: ObstacleKind;
  readonly lane: number;
  readonly width: number;
  readonly height: number;
  /** Cosmetic colour index for cars. */
  readonly variant: number;
  /** Top edge in world px; grows as the obstacle moves down the screen. */
  y: number;
  /** Set once the obstacle is fully behind the player, so it scores once. */
  passed: boolean;
}

/** Bounds of an obstacle in world px. */
export function obstacleRect(o: Obstacle): Rect {
  return { x: laneCenterX(o.lane) - o.width / 2, y: o.y, width: o.width, height: o.height };
}

const lerp = (from: number, to: number, t: number): number => from + (to - from) * t;

/** Lane numbers 0..laneCount-1. */
const laneIndices = (): number[] => Array.from({ length: ROAD.laneCount }, (_, i) => i);

/** Re-roll attempts before falling back to a single obstacle. */
const MAX_ROW_ATTEMPTS = 6;

interface RowPlan {
  readonly lane: number;
  readonly kind: ObstacleKind;
}

/**
 * Spawns, moves and tests obstacles.
 *
 * Rows are spawned by DISTANCE (not wall-clock) so spacing stays consistent
 * when the frame rate varies, and the pixel gap is derived from a TIME gap so
 * the player always gets a similar reaction window as speed ramps up.
 */
export class ObstacleField {
  private readonly rng: Rng;
  private obstacles: Obstacle[] = [];
  private nextId = 1;
  private spawnCountdownPx: number = OBSTACLES.firstRowDistancePx;
  /** Lanes free of cars in the most recent row = where the player may be standing. */
  private previousFreeLanes: readonly number[] = [MIDDLE_LANE];

  constructor(rng: Rng = Math.random) {
    this.rng = rng;
  }

  get items(): readonly Obstacle[] {
    return this.obstacles;
  }

  reset(): void {
    this.obstacles = [];
    this.nextId = 1;
    this.spawnCountdownPx = OBSTACLES.firstRowDistancePx;
    this.previousFreeLanes = [MIDDLE_LANE];
  }

  /**
   * Moves obstacles with the world, spawns new rows and culls old ones.
   * @returns how many obstacles were newly passed this step (for score bonus)
   */
  update(dt: number, road: Road): number {
    const dy = road.speed * dt;
    for (const o of this.obstacles) o.y += dy;

    this.spawnCountdownPx -= dy;
    if (this.spawnCountdownPx <= 0) {
      this.spawnRow(road);
      const gapSeconds = lerp(OBSTACLES.gapSecondsStart, OBSTACLES.gapSecondsMin, road.difficulty);
      this.spawnCountdownPx = gapSeconds * road.speed;
    }

    let newlyPassed = 0;
    const carBottom = CAR.y + CAR.height;
    for (const o of this.obstacles) {
      if (!o.passed && o.y > carBottom) {
        o.passed = true;
        newlyPassed += 1;
      }
    }
    this.obstacles = this.obstacles.filter((o) => o.y < WORLD.height + OBSTACLES.despawnMarginPx);
    return newlyPassed;
  }

  /**
   * The first obstacle the car is currently hitting, or null.
   * Barriers are skipped while the car is airborne; cars never are.
   */
  firstHit(car: Car): Obstacle | null {
    const carRect = car.rect;
    for (const o of this.obstacles) {
      if (o.kind === 'barrier' && car.airborne) continue;
      if (collides(carRect, obstacleRect(o))) return o;
    }
    return null;
  }

  /**
   * Spawns one row just above the screen.
   *
   * FAIRNESS (two rules, both enforced here and checked by tests):
   *  1. A row occupies 1 lane, or 2 with a probability that grows with
   *     difficulty, so at least one lane is always free of obstacles.
   *  2. REACHABILITY: wherever the player could be standing after the previous
   *     row (any lane that had no car), the new row must leave a free lane at
   *     most ONE lane away. Without this, a double row can demand a two-lane
   *     dodge from the screen edge that is physically impossible once camera
   *     latency, the smoother's cooldown and human reaction time are added.
   *     Barrier lanes count as free: a barrier is cleared by a jump, which is
   *     available from any lane.
   */
  private spawnRow(road: Road): void {
    const doubleChance = OBSTACLES.doubleBlockChanceMax * road.difficulty;
    const wantsDouble = this.rng() < doubleChance;

    let row: RowPlan[] = this.planRow(wantsDouble ? 2 : 1);
    for (let attempt = 0; wantsDouble && !this.isReachable(row) && attempt < MAX_ROW_ATTEMPTS; attempt += 1) {
      row = this.planRow(2);
    }
    // A single obstacle is always reachable, so this fallback cannot trap the player.
    if (!this.isReachable(row)) row = this.planRow(1);

    for (const { lane, kind } of row) {
      const isBarrier = kind === 'barrier';
      const height = isBarrier ? OBSTACLES.barrierHeight : OBSTACLES.carHeight;
      this.obstacles.push({
        id: this.nextId,
        kind,
        lane,
        width: isBarrier ? OBSTACLES.barrierWidth : OBSTACLES.carWidth,
        height,
        variant: Math.floor(this.rng() * OBSTACLES.carVariants),
        y: -height,
        passed: false,
      });
      this.nextId += 1;
    }
    this.previousFreeLanes = laneIndices().filter(
      (lane) => !row.some((entry) => entry.lane === lane && entry.kind === 'car'),
    );
  }

  /** Picks `count` distinct lanes (Fisher-Yates with the injected RNG) and a kind for each. */
  private planRow(count: number): RowPlan[] {
    const lanes = laneIndices();
    for (let i = lanes.length - 1; i > 0; i -= 1) {
      const j = Math.floor(this.rng() * (i + 1));
      const a = lanes[i];
      const b = lanes[j];
      if (a !== undefined && b !== undefined) {
        lanes[i] = b;
        lanes[j] = a;
      }
    }
    return lanes.slice(0, count).map((lane) => ({
      lane,
      kind: this.rng() < OBSTACLES.barrierChance ? 'barrier' : 'car',
    }));
  }

  /** Rule 2 above: every possible player lane has a free lane within one step. */
  private isReachable(row: readonly RowPlan[]): boolean {
    const free = laneIndices().filter(
      (lane) => !row.some((entry) => entry.lane === lane && entry.kind === 'car'),
    );
    return this.previousFreeLanes.every((from) => free.some((to) => Math.abs(from - to) <= 1));
  }
}
