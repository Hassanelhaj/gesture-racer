import { HAND_TRACKING, UI } from './constants';
import { AudioEngine } from './game/audio';
import { Game } from './game/game';
import type { GameEvent, GameResult, HudState } from './game/game';
import { GameLoop } from './game/loop';
import { renderGame, syncCanvasResolution } from './game/render';
import { createLogger } from './logger';
import { loadBestScore, saveBestScore } from './storage/bestScore';
import { loadSettings, saveSettings } from './storage/settings';
import type { Settings } from './storage/settings';
import { GESTURES, VISION_ERROR_MESSAGES, VisionError } from './types';
import type { Gesture, VisionErrorCode } from './types';
import { preloadVisionModels } from './vision/loader';
import { createVisionPipeline } from './vision/pipeline';
import type { PipelineStatus, VisionFactory, VisionSource } from './vision/pipeline';

const log = createLogger('session');

export type Screen = 'start' | 'loading' | 'practice' | 'playing' | 'gameover' | 'error';
export type InputMode = 'camera' | 'keyboard';

export interface ErrorInfo {
  readonly code: VisionErrorCode;
  readonly message: string;
}

/** Everything the React tree renders from. Replaced (never mutated) on every change. */
export interface SessionSnapshot {
  readonly screen: Screen;
  readonly inputMode: InputMode | null;
  readonly loadingMessage: string;
  /** 0..1 while models are loading, or null when unknown. */
  readonly loadingFraction: number | null;
  readonly error: ErrorInfo | null;
  readonly hud: HudState;
  readonly best: number;
  readonly newBest: boolean;
  readonly result: GameResult | null;
  /** The run is frozen because the hand left the frame. */
  readonly handPaused: boolean;
  /** Gestures the player has performed during warm-up. */
  readonly practiceDone: Readonly<Record<Gesture, boolean>>;
  readonly vision: PipelineStatus | null;
  readonly settings: Settings;
  /** Rendering smoothness over the last couple of seconds, for the diagnostics panel. */
  readonly perf: FramePerf;
}

/** Frame-rate summary. p95 is the slowest frame time among the best 95 % of frames. */
export interface FramePerf {
  readonly fps: number;
  readonly p95FrameMs: number;
}

/** Number of recent frames kept for the FPS readout (about 2 s at 60 fps). */
const PERF_WINDOW = 120;

function framePerf(times: readonly number[]): FramePerf {
  if (times.length < 10) return { fps: 0, p95FrameMs: 0 };
  const sorted = [...times].sort((a, b) => a - b);
  const mean = times.reduce((sum, t) => sum + t, 0) / times.length;
  return { fps: 1000 / mean, p95FrameMs: sorted[Math.floor(sorted.length * 0.95)] ?? 0 };
}

export interface SessionOptions {
  /** Swap the camera/ML pipeline, e.g. for a scripted demo or a test. */
  readonly createVision?: VisionFactory;
}

const KEY_ACTIONS: Readonly<Record<string, Gesture>> = {
  ArrowLeft: 'LEFT',
  KeyA: 'LEFT',
  ArrowRight: 'RIGHT',
  KeyD: 'RIGHT',
  ArrowUp: 'JUMP',
  KeyW: 'JUMP',
  Space: 'JUMP',
};

const EMPTY_PRACTICE: Readonly<Record<Gesture, boolean>> = { LEFT: false, RIGHT: false, JUMP: false };

/**
 * Orchestrates one play session: screens, camera pipeline, game, audio,
 * persistence and the render loop.
 *
 * WHY a plain class instead of React state: the 60 fps loop and the camera
 * callbacks live outside React's render cycle. This object owns all mutable
 * state and publishes immutable snapshots through subscribe/getSnapshot, so
 * React (via useSyncExternalStore) only re-renders when something the UI shows
 * actually changed (screen switches, plus a throttled ~10 Hz score/vision refresh).
 */
export class GameSession {
  readonly game: Game;
  private readonly audio = new AudioEngine();
  private readonly createVision: VisionFactory;
  private readonly listeners = new Set<() => void>();

  private video: HTMLVideoElement | null = null;
  private context: CanvasRenderingContext2D | null = null;
  private vision: VisionSource | null = null;

  private screen: Screen = 'start';
  private inputMode: InputMode | null = null;
  private loadingMessage = '';
  private loadingFraction: number | null = null;
  private readonly frameTimes: number[] = [];
  private error: ErrorInfo | null = null;
  private best = 0;
  private newBest = false;
  private result: GameResult | null = null;
  private handPaused = false;
  private practiceDone: Readonly<Record<Gesture, boolean>> = EMPTY_PRACTICE;
  private settings: Settings;

  private gameOverTimer: ReturnType<typeof setTimeout> | null = null;
  private lastUiRefreshMs = 0;
  /** Bumped on every camera start/stop so a slow, abandoned start cannot resurrect itself. */
  private visionEpoch = 0;

  private snapshot: SessionSnapshot;

  constructor(options: SessionOptions = {}) {
    this.createVision = options.createVision ?? createVisionPipeline;
    this.settings = loadSettings();
    this.game = new Game({
      onEvent: (event) => {
        this.onGameEvent(event);
      },
      onGameOver: (result) => {
        this.onGameOver(result);
      },
    });
    this.audio.setMuted(!this.settings.sound);
    this.snapshot = this.buildSnapshot();
    void loadBestScore().then((best) => {
      this.best = best;
      this.emit();
    });
  }

  /* ----------------------------- React bridge ----------------------------- */

  readonly subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  readonly getSnapshot = (): SessionSnapshot => this.snapshot;

  /**
   * Connects the session to the DOM and starts the loop.
   * @returns a cleanup that stops everything, safe to call and re-attach
   *          (React StrictMode mounts, unmounts and mounts again in dev).
   */
  attach(canvas: HTMLCanvasElement, video: HTMLVideoElement): () => void {
    this.video = video;
    // alpha:false lets the browser skip blending the canvas with the page: the game paints every pixel anyway.
    this.context = canvas.getContext('2d', { alpha: false });
    if (this.context === null) {
      this.fail(new VisionError('tracking-failed', 'Canvas 2D is not available'));
    }

    const resize = (): void => {
      syncCanvasResolution(canvas);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    window.addEventListener('keydown', this.handleKeyDown);

    const loop = new GameLoop(this.tick);
    loop.start();
    // Warm the models in the background so "Use my camera" is usually instant.
    if (this.settings.preload) preloadVisionModels();

    return () => {
      loop.stop();
      observer.disconnect();
      window.removeEventListener('keydown', this.handleKeyDown);
      this.clearGameOverTimer();
      void this.releaseVision();
      void this.audio.stop();
      this.video = null;
      this.context = null;
    };
  }

  /* -------------------------------- actions ------------------------------- */

  /** Starts (or resumes) camera control: permission, models, then the warm-up screen. */
  async startCamera(): Promise<void> {
    if (this.screen === 'loading' || this.video === null) return;
    void this.audio.start(); // inside the click handler, so the browser allows sound
    this.inputMode = 'camera';
    this.error = null;
    this.practiceDone = EMPTY_PRACTICE;

    if (this.vision !== null) {
      this.vision.reset();
      this.setScreen('practice');
      return;
    }

    const epoch = (this.visionEpoch += 1);
    this.loadingMessage = 'Starting camera…';
    this.loadingFraction = null;
    this.setScreen('loading');
    try {
      const vision = await this.createVision(
        this.video,
        {
          onGesture: (gesture) => {
            this.onVisionGesture(gesture);
          },
          onError: (error) => {
            this.fail(error);
          },
        },
        (message, fraction) => {
          this.loadingMessage = message;
          this.loadingFraction = fraction;
          this.emit();
        },
      );
      if (epoch !== this.visionEpoch) {
        await vision.dispose(); // the player backed out while we were loading
        return;
      }
      this.vision = vision;
      this.setScreen('practice');
    } catch (error) {
      if (epoch === this.visionEpoch) this.fail(error);
    }
  }

  /** Starts a run using keyboard / touch controls (no camera). */
  startKeyboard(): void {
    void this.audio.start();
    this.inputMode = 'keyboard';
    this.startRun();
  }

  /** Begins a run with whatever input mode is active. */
  startRun(): void {
    this.clearGameOverTimer();
    this.game.start();
    this.vision?.reset();
    this.newBest = false;
    this.result = null;
    this.handPaused = false;
    this.error = null;
    this.audio.setEngineOn(true);
    this.setScreen('playing');
  }

  /** Back to the title screen. Also switches the camera OFF: nothing records while idle. */
  async toMenu(): Promise<void> {
    this.clearGameOverTimer();
    this.game.returnToReady();
    this.audio.setEngineOn(false);
    this.handPaused = false;
    this.inputMode = null;
    this.error = null;
    this.setScreen('start');
    await this.releaseVision();
  }

  setSettings(patch: Partial<Settings>): void {
    const wasPreloading = this.settings.preload;
    this.settings = { ...this.settings, ...patch };
    // Turning preload on mid-session starts it right away (turning it off cannot undo a download).
    if (this.settings.preload && !wasPreloading) preloadVisionModels(undefined, 0);
    saveSettings(this.settings);
    this.audio.setMuted(!this.settings.sound);
    this.emit();
  }

  /** Touch fallback: outer thirds steer, the middle third jumps. */
  tap(xFraction: number): void {
    if (this.screen !== 'playing' || this.inputMode !== 'keyboard') return;
    if (xFraction < UI.tapZoneFraction) this.game.apply('LEFT');
    else if (xFraction > 1 - UI.tapZoneFraction) this.game.apply('RIGHT');
    else this.game.apply('JUMP');
  }

  /* ------------------------------ internals ------------------------------- */

  private readonly tick = (dt: number, nowMs: number): void => {
    if (dt > 0) {
      this.frameTimes.push(dt * 1000);
      if (this.frameTimes.length > PERF_WINDOW) this.frameTimes.shift();
    }
    this.vision?.step(nowMs);
    this.updateHandPause(nowMs);
    // A frozen run still redraws (so the picture stays alive) but does not advance.
    if (!(this.screen === 'playing' && this.handPaused)) this.game.update(dt);
    if (this.context !== null) renderGame(this.context, this.game, nowMs);

    if (nowMs - this.lastUiRefreshMs >= UI.refreshMs) {
      this.lastUiRefreshMs = nowMs;
      if (this.screen === 'playing' || this.screen === 'practice' || this.vision !== null) this.emit();
    }
  };

  private updateHandPause(nowMs: number): void {
    const shouldPause =
      this.screen === 'playing' &&
      this.inputMode === 'camera' &&
      this.vision !== null &&
      this.vision.handLostMs(nowMs) > HAND_TRACKING.pauseAfterLostMs;
    if (shouldPause === this.handPaused) return;
    this.handPaused = shouldPause;
    this.audio.setEngineOn(!shouldPause);
    this.emit();
  }

  private readonly handleKeyDown = (event: KeyboardEvent): void => {
    if (event.repeat) return;
    const action = KEY_ACTIONS[event.code];
    if (action !== undefined && this.screen === 'playing') {
      event.preventDefault(); // stop Space/arrows from scrolling the page
      this.game.apply(action);
      return;
    }
    // Enter restarts from the end screen, unless a button has focus (it handles Enter itself).
    if (event.code === 'Enter' && this.screen === 'gameover' && document.activeElement === document.body) {
      this.startRun();
    }
  };

  private onVisionGesture(gesture: Gesture): void {
    if (this.screen === 'playing') {
      if (!this.handPaused) this.game.apply(gesture);
    } else if (this.screen === 'practice' && !this.practiceDone[gesture]) {
      this.practiceDone = { ...this.practiceDone, [gesture]: true };
      this.emit();
    }
  }

  private onGameEvent(event: GameEvent): void {
    switch (event) {
      case 'jump':
        this.audio.playJump();
        break;
      case 'lane':
        this.audio.playLane();
        break;
      case 'pass':
        this.audio.playPass();
        break;
      case 'crash':
        this.audio.playCrash();
        break;
    }
  }

  private onGameOver(result: GameResult): void {
    this.result = result;
    this.newBest = result.score > this.best && result.score > 0;
    void saveBestScore(result.score).then((best) => {
      this.best = best;
      this.emit();
    });
    // Let the crash shake play before the card slides in.
    this.clearGameOverTimer();
    this.gameOverTimer = setTimeout(() => {
      this.gameOverTimer = null;
      if (this.screen === 'playing') this.setScreen('gameover');
    }, UI.gameOverDelayMs);
  }

  /** Any failure ends in the error screen with a player-readable message. */
  private fail(error: unknown): void {
    log.error('session failure', error);
    const code: VisionErrorCode = error instanceof VisionError ? error.code : 'tracking-failed';
    this.error = { code, message: VISION_ERROR_MESSAGES[code] };
    this.clearGameOverTimer();
    this.audio.setEngineOn(false);
    this.handPaused = false;
    this.setScreen('error');
    void this.releaseVision();
  }

  private async releaseVision(): Promise<void> {
    this.visionEpoch += 1;
    const vision = this.vision;
    this.vision = null;
    if (vision !== null) await vision.dispose();
    this.emit();
  }

  private clearGameOverTimer(): void {
    if (this.gameOverTimer !== null) clearTimeout(this.gameOverTimer);
    this.gameOverTimer = null;
  }

  private setScreen(screen: Screen): void {
    this.screen = screen;
    this.emit();
  }

  private emit(): void {
    this.snapshot = this.buildSnapshot();
    for (const listener of this.listeners) listener();
  }

  private buildSnapshot(): SessionSnapshot {
    return {
      screen: this.screen,
      inputMode: this.inputMode,
      loadingMessage: this.loadingMessage,
      loadingFraction: this.loadingFraction,
      error: this.error,
      hud: this.game.hud,
      best: this.best,
      newBest: this.newBest,
      result: this.result,
      handPaused: this.handPaused,
      practiceDone: this.practiceDone,
      vision: this.vision?.status ?? null,
      settings: this.settings,
      perf: framePerf(this.frameTimes),
    };
  }
}

/** Number of gestures, for UI that shows "2 of 3 tried". */
export const GESTURE_COUNT = GESTURES.length;
