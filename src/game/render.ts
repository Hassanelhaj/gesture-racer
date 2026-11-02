import { CAR, FEEDBACK, ROAD, WORLD } from '../constants';
import type { Game } from './game';
import { obstacleRect } from './obstacles';
import type { Obstacle } from './obstacles';
import { DASH_PERIOD, LANE_WIDTH, ROAD_LEFT, ROAD_RIGHT } from './road';

/** Highest device-pixel-ratio rendered; 2x keeps edges crisp without 4x fill cost on phones. */
const MAX_DPR = 2;

/** Night palette, matching the dark UI. */
const PALETTE = {
  grass: '#0b1018',
  grassLight: '#101826',
  asphalt: '#1a202c',
  asphaltLight: '#222a39',
  edge: 'rgba(232, 236, 244, 0.55)',
  dash: 'rgba(232, 236, 244, 0.38)',
  post: '#6ea8ff',
  player: '#ff4d6d',
  playerDark: '#c2253f',
  glass: '#0a0d14',
  headlight: '#fff6d6',
  taillight: '#ff2a2a',
  barrierA: '#ff9f1c',
  barrierB: '#f4f1e8',
  crash: '#ff2a2a',
} as const;

const CAR_COLORS: readonly string[] = ['#4cc9f0', '#f9c74f', '#90be6d', '#b388eb', '#e8ecf4'];

/** Spacing of the roadside lights that give a sense of speed. */
const POST_SPACING = 120;
/**
 * Height of the fade at the top. Kept short and translucent: it also hides
 * the player's earliest warning of an incoming row, so it must not eat much.
 */
const HAZE_HEIGHT = 90;
/** Visual-only fake height of the shadow offset while jumping. */
const SHADOW_JUMP_SHRINK = 0.45;

/**
 * Sizes the canvas backing store to its on-screen size times the device pixel
 * ratio, so drawing is sharp on high-DPI screens. Call on mount and on resize.
 *
 * @returns the scale from world px to canvas pixels
 */
export function syncCanvasResolution(canvas: HTMLCanvasElement): number {
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
  const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.round((width * WORLD.height) / WORLD.width));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return width / WORLD.width;
}

/** Extra world pixels cached around the screen, so the crash shake never reveals an edge. */
const LAYER_PAD = 24;

interface StaticLayers {
  /** Canvas width in device pixels this cache was built for. */
  readonly key: number;
  /** Night grass, asphalt and edge lines: everything that never moves. */
  readonly backdrop: HTMLCanvasElement;
  /** The soft fade at the top of the screen. */
  readonly haze: HTMLCanvasElement;
}

let layers: StaticLayers | null = null;

/** Draws `paint` once, in world units, onto a new canvas `widthWorld` x `heightWorld` big. */
function buildLayer(
  widthWorld: number,
  heightWorld: number,
  scale: number,
  paint: (ctx: CanvasRenderingContext2D) => void,
): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = Math.ceil(widthWorld * scale);
  canvas.height = Math.ceil(heightWorld * scale);
  const ctx = canvas.getContext('2d');
  if (ctx === null) throw new Error('Canvas 2D is not available');
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  paint(ctx);
  return canvas;
}

/**
 * Returns the cached static layers, rebuilding them when the canvas size changes.
 *
 * WHY cache: the gradients and road never change, yet drawing them from scratch
 * every frame (several full-screen gradient fills) was a large share of the
 * per-frame cost on integrated GPUs. One `drawImage` replaces all of it.
 */
function getLayers(scale: number, canvasWidth: number): StaticLayers {
  if (layers?.key === canvasWidth) return layers;
  const backdrop = buildLayer(WORLD.width + 2 * LAYER_PAD, WORLD.height + 2 * LAYER_PAD, scale, (ctx) => {
    ctx.translate(LAYER_PAD, LAYER_PAD); // so world coordinates still work, with padding around them
    paintBackdrop(ctx);
  });
  const haze = buildLayer(WORLD.width, HAZE_HEIGHT, scale, paintHaze);
  layers = { key: canvasWidth, backdrop, haze };
  return layers;
}

/**
 * Draws one frame of the game. Pure drawing: reads game state, never changes it.
 *
 * @param timeMs a monotonic clock, used only for cosmetic wobble
 */
export function renderGame(ctx: CanvasRenderingContext2D, game: Game, timeMs: number): void {
  const scale = ctx.canvas.width / WORLD.width;
  const { backdrop, haze } = getLayers(scale, ctx.canvas.width);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);

  const intensity = game.crashIntensity;
  ctx.save();
  if (intensity > 0) {
    // Decaying shake; sin of different frequencies avoids a visible pattern.
    const amp = FEEDBACK.crashShakePx * intensity;
    ctx.translate(Math.sin(timeMs * 0.09) * amp, Math.cos(timeMs * 0.13) * amp);
  }

  ctx.drawImage(backdrop, -LAYER_PAD, -LAYER_PAD, WORLD.width + 2 * LAYER_PAD, WORLD.height + 2 * LAYER_PAD);
  drawRoadsideLights(ctx, game.road.offset);
  drawLaneDashes(ctx, game.road.offset);
  drawSpeedLines(ctx, game.road.offset, game.road.difficulty);
  for (const obstacle of game.obstacles.items) drawObstacle(ctx, obstacle);
  drawPlayer(ctx, game);
  ctx.restore();

  ctx.drawImage(haze, 0, 0, WORLD.width, HAZE_HEIGHT);
  if (game.phase === 'gameover') {
    const flash = Math.max(0, 1 - game.crashAgeS / FEEDBACK.crashFlashS);
    if (flash > 0) {
      ctx.fillStyle = PALETTE.crash;
      ctx.globalAlpha = flash * 0.35;
      ctx.fillRect(0, 0, WORLD.width, WORLD.height);
      ctx.globalAlpha = 1;
    }
  }
}

/** Static part of the scene: grass, asphalt and the two edge lines. Drawn once into a cache. */
function paintBackdrop(ctx: CanvasRenderingContext2D): void {
  const grass = ctx.createLinearGradient(0, 0, 0, WORLD.height);
  grass.addColorStop(0, PALETTE.grass);
  grass.addColorStop(1, PALETTE.grassLight);
  ctx.fillStyle = grass;
  ctx.fillRect(-LAYER_PAD, -LAYER_PAD, WORLD.width + 2 * LAYER_PAD, WORLD.height + 2 * LAYER_PAD);

  const asphalt = ctx.createLinearGradient(0, 0, 0, WORLD.height);
  asphalt.addColorStop(0, PALETTE.asphalt);
  asphalt.addColorStop(1, PALETTE.asphaltLight);
  ctx.fillStyle = asphalt;
  ctx.fillRect(ROAD_LEFT, -LAYER_PAD, ROAD.width, WORLD.height + 2 * LAYER_PAD);

  ctx.fillStyle = PALETTE.edge;
  ctx.fillRect(ROAD_LEFT - ROAD.edgeLineWidth, -LAYER_PAD, ROAD.edgeLineWidth, WORLD.height + 2 * LAYER_PAD);
  ctx.fillRect(ROAD_RIGHT, -LAYER_PAD, ROAD.edgeLineWidth, WORLD.height + 2 * LAYER_PAD);
}

/** Roadside lights scroll with the world: the strongest cheap cue for speed. */
function drawRoadsideLights(ctx: CanvasRenderingContext2D, offset: number): void {
  ctx.fillStyle = PALETTE.post;
  for (let y = (offset % POST_SPACING) - POST_SPACING; y < WORLD.height + POST_SPACING; y += POST_SPACING) {
    for (const x of [ROAD_LEFT - 22, ROAD_RIGHT + 22]) {
      ctx.globalAlpha = 0.18;
      ctx.beginPath();
      ctx.arc(x, y, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
}

function drawLaneDashes(ctx: CanvasRenderingContext2D, offset: number): void {
  ctx.fillStyle = PALETTE.dash;
  for (let lane = 1; lane < ROAD.laneCount; lane += 1) {
    const x = ROAD_LEFT + LANE_WIDTH * lane - 2;
    for (let y = offset - DASH_PERIOD; y < WORLD.height; y += DASH_PERIOD) {
      ctx.fillRect(x, y, 4, ROAD.dashLength);
    }
  }
}

/** Faint streaks that fade in as the car approaches top speed. */
function drawSpeedLines(ctx: CanvasRenderingContext2D, offset: number, difficulty: number): void {
  if (difficulty <= 0.05) return;
  ctx.fillStyle = '#e8ecf4';
  ctx.globalAlpha = 0.07 * difficulty;
  for (let i = 0; i < 9; i += 1) {
    const x = ROAD_LEFT + 14 + ((i * 97) % (ROAD.width - 28));
    const y = ((offset * 2.2 + i * 151) % (WORLD.height + 160)) - 160;
    ctx.fillRect(x, y, 2, 90 + difficulty * 80);
  }
  ctx.globalAlpha = 1;
}

/** Soft fade at the top of the screen so spawning rows appear out of the dark. Drawn once into a cache. */
function paintHaze(ctx: CanvasRenderingContext2D): void {
  const gradient = ctx.createLinearGradient(0, 0, 0, HAZE_HEIGHT);
  gradient.addColorStop(0, 'rgba(11, 16, 24, 0.7)');
  gradient.addColorStop(1, 'rgba(11, 16, 24, 0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, WORLD.width, HAZE_HEIGHT);
}

/**
 * Draws a top-down car centred on (cx, cy), nose pointing up.
 * Callers flip it vertically for oncoming traffic.
 */
function drawCarBody(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  width: number,
  height: number,
  body: string,
  dark: string,
): void {
  const x = cx - width / 2;
  const y = cy - height / 2;
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, 12);
  ctx.fill();

  // Roof and windscreen.
  ctx.fillStyle = dark;
  ctx.beginPath();
  ctx.roundRect(x + 7, y + height * 0.2, width - 14, height * 0.2, 6);
  ctx.fill();
  ctx.beginPath();
  ctx.roundRect(x + 9, y + height * 0.62, width - 18, height * 0.16, 5);
  ctx.fill();
  ctx.fillStyle = PALETTE.glass;
  ctx.globalAlpha = 0.65;
  ctx.beginPath();
  ctx.roundRect(x + 10, y + height * 0.42, width - 20, height * 0.18, 4);
  ctx.fill();
  ctx.globalAlpha = 1;

  // Headlights at the nose, taillights at the rear.
  ctx.fillStyle = PALETTE.headlight;
  ctx.fillRect(x + 4, y + 3, 11, 5);
  ctx.fillRect(x + width - 15, y + 3, 11, 5);
  ctx.fillStyle = PALETTE.taillight;
  ctx.fillRect(x + 4, y + height - 7, 11, 4);
  ctx.fillRect(x + width - 15, y + height - 7, 11, 4);
}

function drawObstacle(ctx: CanvasRenderingContext2D, obstacle: Obstacle): void {
  const rect = obstacleRect(obstacle);
  if (obstacle.kind === 'barrier') {
    drawBarrier(ctx, rect.x, rect.y, rect.width, rect.height);
    return;
  }
  const cx = rect.x + rect.width / 2;
  const cy = rect.y + rect.height / 2;

  // Headlight pool on the road ahead of the oncoming car (its nose faces down).
  const glow = ctx.createRadialGradient(cx, rect.y + rect.height + 18, 4, cx, rect.y + rect.height + 18, 70);
  glow.addColorStop(0, 'rgba(255, 246, 214, 0.28)');
  glow.addColorStop(1, 'rgba(255, 246, 214, 0)');
  ctx.fillStyle = glow;
  // Cover the gradient's full radius (70) so its edge never shows as a hard line.
  ctx.fillRect(cx - 70, rect.y + rect.height - 52, 140, 140);

  drawShadow(ctx, cx, cy + 6, rect.width, rect.height, 0.35);
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(1, -1);
  const color = CAR_COLORS[obstacle.variant % CAR_COLORS.length] ?? '#4cc9f0';
  drawCarBody(ctx, 0, 0, rect.width, rect.height, color, 'rgba(0, 0, 0, 0.28)');
  ctx.restore();
}

function drawBarrier(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number): void {
  drawShadow(ctx, x + width / 2, y + height / 2 + 4, width, height, 0.4);
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, 6);
  ctx.clip();
  ctx.fillStyle = PALETTE.barrierB;
  ctx.fillRect(x, y, width, height);
  ctx.fillStyle = PALETTE.barrierA;
  for (let sx = x - height; sx < x + width; sx += 28) {
    ctx.beginPath();
    ctx.moveTo(sx, y + height);
    ctx.lineTo(sx + 14, y + height);
    ctx.lineTo(sx + 14 + height, y);
    ctx.lineTo(sx + height, y);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
}

function drawShadow(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  width: number,
  height: number,
  alpha: number,
): void {
  ctx.fillStyle = `rgba(0, 0, 0, ${alpha})`;
  ctx.beginPath();
  ctx.ellipse(cx, cy, width * 0.55, height * 0.52, 0, 0, Math.PI * 2);
  ctx.fill();
}

function drawPlayer(ctx: CanvasRenderingContext2D, game: Game): void {
  const { car } = game;
  const rect = car.rect;
  const cx = rect.x + rect.width / 2;
  const groundY = rect.y + rect.height / 2;
  const h = car.height01;

  // The shadow stays on the ground and shrinks/fades as the car rises.
  drawShadow(ctx, cx, groundY + 8, rect.width * (1 - SHADOW_JUMP_SHRINK * h), rect.height, 0.45 * (1 - 0.6 * h));

  // Fake height in a top-down view: scale up and lift.
  const scale = 1 + (CAR.jumpPeakScale - 1) * h;
  ctx.save();
  ctx.translate(cx, groundY - h * CAR.jumpLiftPx);
  ctx.scale(scale, scale);
  if (game.phase === 'gameover') ctx.rotate(Math.min(game.crashAgeS * 3, 0.5));
  drawCarBody(ctx, 0, 0, rect.width, rect.height, PALETTE.player, PALETTE.playerDark);
  ctx.restore();
}
