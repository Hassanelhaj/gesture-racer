import type { CSSProperties } from 'react';
import { GESTURES } from '../types';
import type { Gesture } from '../types';
import type { PipelineStatus } from '../vision/pipeline';
import { GESTURE_INFO, GestureGlyph } from './gestures';

/** Simple cog: a ring with eight rounded teeth. Stroke-based so it stays crisp at any size. */
function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" width="100%" height="100%" aria-hidden="true" fill="none" stroke="currentColor" strokeLinecap="round">
      <circle cx="12" cy="12" r="4.2" strokeWidth="2.6" />
      {[0, 45, 90, 135, 180, 225, 270, 315].map((angle) => (
        <line key={angle} x1="12" y1="2.6" x2="12" y2="5.2" strokeWidth="3" transform={`rotate(${angle} 12 12)`} />
      ))}
    </svg>
  );
}

export { GearIcon };

interface StartProps {
  readonly best: number;
  readonly onCamera: () => void;
  readonly onKeyboard: () => void;
  readonly onSettings: () => void;
}

/** Title screen: what the game is, the three gestures, and how to begin. */
export function StartScreen({ best, onCamera, onKeyboard, onSettings }: StartProps) {
  return (
    <section className="screen start" aria-labelledby="game-title">
      <button type="button" className="icon-btn corner" onClick={onSettings} aria-label="Settings">
        <GearIcon />
      </button>

      <h1 id="game-title" className="logo">
        <span className="logo-top">Gesture</span>
        <span className="logo-bottom">Racer</span>
      </h1>
      <p className="tagline">Drive with your bare hand. No controller, no cloud.</p>

      <ul className="gesture-cards" aria-label="The three gestures">
        {GESTURES.map((gesture, index) => {
          const info = GESTURE_INFO[gesture];
          return (
            <li
              key={gesture}
              className="gesture-card clay"
              style={{ '--c1': info.colors[0], '--c2': info.colors[1], '--delay': `${index * 120}ms` } as CSSProperties}
            >
              <GestureGlyph gesture={gesture} />
              <strong>{info.title}</strong>
              <small>{info.hint}</small>
            </li>
          );
        })}
      </ul>

      <div className="actions">
        <button type="button" className="clay-btn primary big" onClick={onCamera}>
          <span aria-hidden="true">📷</span> Use my camera
        </button>
        <button type="button" className="clay-btn soft" onClick={onKeyboard}>
          <span aria-hidden="true">⌨️</span> Keyboard or touch
        </button>
      </div>

      <p className="pill privacy">
        <span aria-hidden="true">🔒</span> Runs on your device. Video never leaves it.
      </p>
      {best > 0 && (
        <p className="pill best-pill">
          <span aria-hidden="true">🏆</span> Best <strong>{best.toLocaleString()}</strong>
        </p>
      )}
    </section>
  );
}

interface PracticeProps {
  readonly status: PipelineStatus | null;
  readonly done: Readonly<Record<Gesture, boolean>>;
  readonly onStart: () => void;
  readonly onSettings: () => void;
}

/**
 * Live warm-up. Each card lights up while its gesture is recognised and keeps
 * a tick once tried, so the player learns the poses (and sees that tracking
 * works) before the first crash.
 */
export function PracticeScreen({ status, done, onStart, onSettings }: PracticeProps) {
  const doneCount = GESTURES.filter((gesture) => done[gesture]).length;
  const message = status === null || !status.handPresent
    ? 'Hold your hand up to the camera'
    : doneCount === GESTURES.length
      ? 'Nice! You are ready'
      : 'Try each gesture';

  return (
    <section className="screen practice" aria-labelledby="practice-title">
      <button type="button" className="icon-btn corner" onClick={onSettings} aria-label="Settings">
        <GearIcon />
      </button>
      <h2 id="practice-title" className="screen-title">Warm up</h2>
      <p className="tagline" role="status">{message}</p>

      <ul className="gesture-cards compact" aria-label="Gestures to try">
        {GESTURES.map((gesture) => {
          const info = GESTURE_INFO[gesture];
          const active = status?.stableGesture === gesture;
          return (
            <li
              key={gesture}
              className={`gesture-card clay${active ? ' active' : ''}${done[gesture] ? ' done' : ''}`}
              style={{ '--c1': info.colors[0], '--c2': info.colors[1] } as CSSProperties}
            >
              <GestureGlyph gesture={gesture} />
              <strong>{info.title}</strong>
              <span className="tick" aria-label={done[gesture] ? 'done' : 'not tried yet'}>
                {done[gesture] ? '✓' : ''}
              </span>
            </li>
          );
        })}
      </ul>

      {status?.classifier === 'rules' && (
        <p className="pill">Built-in rules mode. Train a model for best accuracy.</p>
      )}

      <div className="actions">
        <button type="button" className="clay-btn primary big" onClick={onStart}>
          Start driving
        </button>
      </div>
    </section>
  );
}
