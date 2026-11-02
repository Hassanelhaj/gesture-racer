import { LOOP } from '../constants';

/** Abstraction over requestAnimationFrame so the loop can be tested without a browser. */
export interface FrameScheduler {
  request: (callback: (timeMs: number) => void) => number;
  cancel: (handle: number) => void;
}

const BROWSER_SCHEDULER: FrameScheduler = {
  request: (callback) => requestAnimationFrame(callback),
  cancel: (handle) => {
    cancelAnimationFrame(handle);
  },
};

/** Called once per frame with the clamped delta (seconds) and the raw timestamp (ms). */
export type TickHandler = (deltaS: number, nowMs: number) => void;

/**
 * requestAnimationFrame loop with delta-time.
 *
 * WHY clamp the delta: when a tab is backgrounded rAF stops, and the first
 * frame after returning would report seconds of elapsed time. Feeding that to
 * the simulation would teleport the car through obstacles, so the step is
 * capped (LOOP.maxDeltaS). The game simply "pauses" while hidden.
 */
export class GameLoop {
  private readonly tick: TickHandler;
  private readonly scheduler: FrameScheduler;
  private handle: number | null = null;
  private lastMs: number | null = null;

  constructor(tick: TickHandler, scheduler: FrameScheduler = BROWSER_SCHEDULER) {
    this.tick = tick;
    this.scheduler = scheduler;
  }

  get running(): boolean {
    return this.handle !== null;
  }

  /** Starts the loop. Calling it twice is a no-op. */
  start(): void {
    if (this.handle !== null) return;
    this.lastMs = null;
    this.handle = this.scheduler.request(this.frame);
  }

  stop(): void {
    if (this.handle !== null) this.scheduler.cancel(this.handle);
    this.handle = null;
    this.lastMs = null;
  }

  private readonly frame = (nowMs: number): void => {
    // The first frame has no previous timestamp, so it advances by 0.
    const rawDeltaS = this.lastMs === null ? 0 : (nowMs - this.lastMs) / 1000;
    this.lastMs = nowMs;
    const deltaS = Math.min(Math.max(rawDeltaS, 0), LOOP.maxDeltaS);

    // Schedule the next frame before ticking so one throwing tick cannot halt
    // the loop silently; callers that want to stop on error call stop().
    this.handle = this.scheduler.request(this.frame);
    this.tick(deltaS, nowMs);
  };
}
