import { useEffect, useState } from 'react';
import type { CSSProperties } from 'react';
import type { GameResult } from '../game/game';
import { GearIcon } from './StartScreen';

interface GameOverProps {
  readonly result: GameResult;
  readonly best: number;
  readonly isNewBest: boolean;
  readonly onRetry: () => void;
  readonly onMenu: () => void;
  readonly onSettings: () => void;
}

const COUNT_UP_MS = 700;

/** Eases the displayed score from 0 to `target`, honouring reduced-motion. */
function useCountUp(target: number): number {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      const id = requestAnimationFrame(() => {
        setValue(target);
      });
      return () => {
        cancelAnimationFrame(id);
      };
    }
    let frame = 0;
    const startedAt = performance.now();
    const step = (now: number): void => {
      const t = Math.min(1, (now - startedAt) / COUNT_UP_MS);
      setValue(Math.round(target * (1 - Math.pow(1 - t, 3))));
      if (t < 1) frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(frame);
    };
  }, [target]);
  return value;
}

/** End-of-run card: the score counts up, a new record gets a badge. */
export function GameOver({ result, best, isNewBest, onRetry, onMenu, onSettings }: GameOverProps) {
  const shown = useCountUp(result.score);
  return (
    <section className="screen gameover" aria-labelledby="over-title">
      <button type="button" className="icon-btn corner" onClick={onSettings} aria-label="Settings">
        <GearIcon />
      </button>

      <div className="clay card over-card" style={{ '--c1': '#4b4390', '--c2': '#2b2760' } as CSSProperties}>
        <h2 id="over-title" className="screen-title">Crash!</h2>
        {isNewBest && <p className="badge">New best!</p>}
        <p className="big-score" aria-label={`Score ${result.score}`}>{shown.toLocaleString()}</p>
        <dl className="over-stats">
          <div>
            <dt>Distance</dt>
            <dd>{result.meters.toLocaleString()} m</dd>
          </div>
          <div>
            <dt>Best</dt>
            <dd>{Math.max(best, result.score).toLocaleString()}</dd>
          </div>
        </dl>
      </div>

      <div className="actions">
        <button type="button" className="clay-btn primary big" onClick={onRetry} autoFocus>
          Drive again
        </button>
        <button type="button" className="clay-btn soft" onClick={onMenu}>
          Menu
        </button>
      </div>
    </section>
  );
}
