/**
 * Development-only data collector for training the gesture classifier.
 * Served by `npm run collect` at /ml/collect.html; it is not part of the
 * production bundle (nothing in the app imports this file).
 *
 * WHY it reuses extractFeatures: training data must be computed by the exact
 * code that runs in the game. Re-implementing the features in Python would let
 * the two drift apart silently and ruin accuracy without any error.
 *
 * Privacy: only 12 numbers per frame are kept. No pixels are stored.
 */
import { GESTURES } from '../types';
import type { Gesture } from '../types';
import { FEATURE_NAMES, extractFeatures } from '../vision/features';
import { HandTracker, openCamera, videoAspectRatio } from '../vision/hands';
import type { CameraHandle } from '../vision/hands';

/** Frames kept per take (about 2 s at 30 fps). */
const TAKE_FRAMES = 60;
/** Per-gesture frame goal shown in the table (from the project brief). */
const TARGET_FRAMES = 300;
const COUNTDOWN_SECONDS = 3;
/** A take is discarded if the hand disappears for longer than this. */
const MAX_HAND_GAP_MS = 1500;

interface Sample {
  readonly gesture: Gesture;
  readonly group: string;
  readonly features: Float32Array;
}

function byId(id: string): HTMLElement {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Missing #${id}`);
  return element;
}

const video = byId('cam') as HTMLVideoElement;
const startButton = byId('start') as HTMLButtonElement;
const recordButton = byId('record') as HTMLButtonElement;
const undoButton = byId('undo') as HTMLButtonElement;
const downloadButton = byId('download') as HTMLButtonElement;
const statusEl = byId('status');
const liveEl = byId('live');
const countdownEl = byId('countdown');
const progressEl = byId('progress') as HTMLProgressElement;
const countsEl = byId('counts');
const gestureButtons = Array.from(document.querySelectorAll<HTMLButtonElement>('#gestures button'));

/** Unique per page load, so groups from different sessions never collide in one CSV. */
const sessionId = Date.now().toString(36);
const samples: Sample[] = [];
const takeCounters: Record<Gesture, number> = { LEFT: 0, RIGHT: 0, JUMP: 0 };
let selected: Gesture = 'LEFT';
let tracker: HandTracker | null = null;
let camera: CameraHandle | null = null;

type Phase = { kind: 'idle' } | { kind: 'countdown' } | { kind: 'recording'; group: string; frames: Sample[]; lastHandMs: number };
let phase: Phase = { kind: 'idle' };

function setStatus(message: string): void {
  statusEl.textContent = message;
}

function renderCounts(): void {
  countsEl.replaceChildren();
  for (const gesture of GESTURES) {
    const mine = samples.filter((s) => s.gesture === gesture);
    const takes = new Set(mine.map((s) => s.group)).size;
    const row = document.createElement('tr');
    const done = mine.length >= TARGET_FRAMES && takes >= 6;
    for (const text of [gesture, String(mine.length), String(takes), done ? 'done' : `${Math.max(0, TARGET_FRAMES - mine.length)} to go`]) {
      const cell = document.createElement('td');
      cell.textContent = text;
      row.append(cell);
    }
    countsEl.append(row);
  }
  downloadButton.disabled = samples.length === 0;
  undoButton.disabled = samples.length === 0 || phase.kind !== 'idle';
}

for (const button of gestureButtons) {
  button.addEventListener('click', () => {
    selected = button.dataset.gesture as Gesture;
    for (const other of gestureButtons) other.classList.toggle('sel', other === button);
  });
}

startButton.addEventListener('click', () => {
  startButton.disabled = true;
  setStatus('Starting camera and hand model…');
  void (async () => {
    try {
      camera = await openCamera(video);
      tracker = await HandTracker.create();
      recordButton.disabled = false;
      setStatus(`Ready (hand model on ${tracker.delegate}). Pick a gesture and press Record.`);
      requestAnimationFrame(loop);
    } catch (error) {
      setStatus(error instanceof Error ? `Failed: ${error.message}` : 'Failed to start');
      startButton.disabled = false;
    }
  })();
});

recordButton.addEventListener('click', () => {
  if (phase.kind !== 'idle') return;
  phase = { kind: 'countdown' };
  recordButton.disabled = true;
  undoButton.disabled = true;
  let remaining = COUNTDOWN_SECONDS;
  countdownEl.textContent = String(remaining);
  const timer = setInterval(() => {
    remaining -= 1;
    if (remaining > 0) {
      countdownEl.textContent = String(remaining);
      return;
    }
    clearInterval(timer);
    countdownEl.textContent = '●';
    takeCounters[selected] += 1;
    phase = { kind: 'recording', group: `${sessionId}-${selected}${takeCounters[selected]}`, frames: [], lastHandMs: performance.now() };
    setStatus(`Recording ${selected}… hold the pose and move slightly.`);
  }, 1000);
});

undoButton.addEventListener('click', () => {
  const last = samples.at(-1);
  if (last === undefined) return;
  for (let i = samples.length - 1; i >= 0 && samples[i]?.group === last.group; i -= 1) samples.pop();
  setStatus(`Removed take ${last.group}.`);
  renderCounts();
});

downloadButton.addEventListener('click', () => {
  const header = ['gesture', 'group', ...FEATURE_NAMES].join(',');
  const rows = samples.map((s) => [s.gesture, s.group, ...Array.from(s.features, (v) => v.toFixed(6))].join(','));
  const blob = new Blob([[header, ...rows].join('\n'), '\n'], { type: 'text/csv' });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `gestures-${sessionId}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
  setStatus(`Saved ${samples.length} frames. Move the file into ml/data/.`);
});

window.addEventListener('beforeunload', (event) => {
  if (samples.length > 0) event.preventDefault(); // recordings live only in memory until downloaded
});

function finishTake(frames: Sample[]): void {
  samples.push(...frames);
  phase = { kind: 'idle' };
  countdownEl.textContent = '';
  recordButton.disabled = false;
  progressEl.value = 0;
  setStatus(`Take saved (${frames.length} frames). Change distance, position or tilt, then record again.`);
  renderCounts();
}

function abortTake(reason: string): void {
  phase = { kind: 'idle' };
  countdownEl.textContent = '';
  recordButton.disabled = false;
  progressEl.value = 0;
  setStatus(`Take discarded: ${reason}`);
  renderCounts();
}

function loop(now: number): void {
  requestAnimationFrame(loop);
  if (tracker === null) return;
  const result = tracker.detect(video, now);
  if (result.kind === 'stale') return;

  if (result.kind === 'no-hand') {
    liveEl.textContent = 'no hand';
    if (phase.kind === 'recording' && now - phase.lastHandMs > MAX_HAND_GAP_MS) abortTake('hand left the frame');
    return;
  }
  const features = extractFeatures(result.frame.landmarks, videoAspectRatio(video));
  if (features === null) {
    liveEl.textContent = 'hand geometry unusable (palm facing the camera head-on?)';
    return;
  }
  liveEl.textContent = FEATURE_NAMES.map((name, i) => `${name.padEnd(14)} ${(features[i] ?? 0).toFixed(3)}`).join('\n');

  if (phase.kind !== 'recording') return;
  phase.lastHandMs = now;
  phase.frames.push({ gesture: selected, group: phase.group, features });
  progressEl.value = phase.frames.length / TAKE_FRAMES;
  if (phase.frames.length >= TAKE_FRAMES) finishTake(phase.frames);
}

window.addEventListener('pagehide', () => {
  camera?.stop();
  tracker?.close();
});

renderCounts();
