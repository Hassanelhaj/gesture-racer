import { FEEDBACK } from '../constants';
import type { Gesture } from '../types';
import { Car } from './car';
import { ObstacleField } from './obstacles';
import type { Rng } from './obstacles';
import { Road } from './road';
import { ScoreKeeper, speedToKmh } from './score';

export type GamePhase = 'ready' | 'playing' | 'gameover';

/** Things the UI/audio may want to react to. */
export type GameEvent = 'lane' | 'jump' | 'pass' | 'crash';

export interface GameResult {
  readonly score: number;
  readonly meters: number;
}

export interface GameOptions {
  /** Injected for deterministic tests; defaults to Math.random. */
  readonly rng?: Rng;
  readonly onEvent?: (event: GameEvent) => void;
  readonly onGameOver?: (result: GameResult) => void;
}

/** Numbers the HUD shows. */
export interface HudState {
  readonly score: number;
  readonly speedKmh: number;
  readonly difficulty: number;
}

/**
 * Owns all gameplay state and the ready -> playing -> gameover state machine.
 *
 * It knows nothing about the camera, canvas or DOM: input arrives as a
 * Gesture via `apply`, and time as `update(dt)`. That separation keeps it
 * unit-testable and lets the keyboard fallback and the hand tracker share the
 * exact same control path.
 */
export class Game {
  readonly road = new Road();
  readonly car = new Car();
  readonly obstacles: ObstacleField;
  readonly score = new ScoreKeeper();

  phase: GamePhase = 'ready';
  /** Seconds since the crash; drives shake/flash. Meaningful in 'gameover'. */
  crashAgeS = 0;
  /** Result of the last finished run, or null before the first crash. */
  lastResult: GameResult | null = null;

  private readonly onEvent: ((event: GameEvent) => void) | undefined;
  private readonly onGameOver: ((result: GameResult) => void) | undefined;

  constructor(options: GameOptions = {}) {
    this.obstacles = new ObstacleField(options.rng);
    this.onEvent = options.onEvent;
    this.onGameOver = options.onGameOver;
  }

  /** Resets everything and begins a run. Safe to call from any phase. */
  start(): void {
    this.road.reset();
    this.car.reset();
    this.obstacles.reset();
    this.score.reset();
    this.crashAgeS = 0;
    this.phase = 'playing';
  }

  /** Clears the run and returns to the idle 'ready' state (menu background). */
  returnToReady(): void {
    this.start();
    this.phase = 'ready';
  }

  /**
   * Applies a control input. Ignored unless a run is in progress, so stray
   * gestures on the menu or game-over screen cannot move anything.
   */
  apply(action: Gesture): void {
    if (this.phase !== 'playing') return;
    if (action === 'JUMP') {
      if (this.car.jump()) this.onEvent?.('jump');
      return;
    }
    if (this.car.steer(action === 'LEFT' ? -1 : 1)) this.onEvent?.('lane');
  }

  /** Advances the simulation by `dt` seconds (already clamped by the loop). */
  update(dt: number): void {
    switch (this.phase) {
      case 'ready':
        // Idle scroll so the menu background feels alive.
        this.road.scroll(dt);
        break;
      case 'gameover':
        this.crashAgeS += dt;
        break;
      case 'playing':
        this.stepPlaying(dt);
        break;
    }
  }

  /** Values for the HUD. */
  get hud(): HudState {
    return {
      score: this.score.value,
      speedKmh: speedToKmh(this.road.speed),
      difficulty: this.road.difficulty,
    };
  }

  /** 0..1 strength of the crash shake/flash, for the renderer. */
  get crashIntensity(): number {
    if (this.phase !== 'gameover') return 0;
    return Math.max(0, 1 - this.crashAgeS / FEEDBACK.crashShakeS);
  }

  private stepPlaying(dt: number): void {
    this.road.update(dt);
    this.car.update(dt);
    this.score.addDistance(this.road.speed * dt);

    const passed = this.obstacles.update(dt, this.road);
    if (passed > 0) {
      this.score.addPassed(passed);
      this.onEvent?.('pass');
    }

    if (this.obstacles.firstHit(this.car) !== null) this.crash();
  }

  private crash(): void {
    this.phase = 'gameover';
    this.crashAgeS = 0;
    this.lastResult = { score: this.score.value, meters: this.score.meters };
    this.onEvent?.('crash');
    this.onGameOver?.(this.lastResult);
  }
}
