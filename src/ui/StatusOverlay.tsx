import type { CSSProperties } from 'react';
interface LoadingProps {
  readonly message: string;
  /** 0..1, or null when the amount is unknown (shows only the spinner). */
  readonly fraction: number | null;
}

/** Camera / model loading: spinner, and a progress bar once the amount is known. */
export function LoadingOverlay({ message, fraction }: LoadingProps) {
  const percent = fraction === null ? null : Math.round(fraction * 100);
  return (
    <section className="screen loading" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <p className="screen-title small">{message}</p>
      {percent !== null && (
        <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} aria-label="Loading progress">
          <div className="progress-fill" style={{ width: `${percent}%` }} />
          <span className="progress-text">{percent}%</span>
        </div>
      )}
      <p className="tagline">The first visit downloads about 22 MB. It is saved on your device, so later visits start much faster.</p>
    </section>
  );
}

interface ErrorProps {
  readonly message: string;
  readonly canRetry: boolean;
  readonly onRetry: () => void;
  readonly onKeyboard: () => void;
  readonly onBack: () => void;
}

/** Friendly failure screen: says what happened and always offers a way to keep playing. */
export function ErrorOverlay({ message, canRetry, onRetry, onKeyboard, onBack }: ErrorProps) {
  return (
    <section className="screen error" role="alert">
      <div className="clay card" style={{ '--c1': '#5a3a63', '--c2': '#352347' } as CSSProperties}>
        <p className="oops" aria-hidden="true">😵</p>
        <h2 className="screen-title small">Something went wrong</h2>
        <p className="error-text">{message}</p>
      </div>
      <div className="actions">
        {canRetry && (
          <button type="button" className="clay-btn primary" onClick={onRetry}>
            Try again
          </button>
        )}
        <button type="button" className="clay-btn soft" onClick={onKeyboard}>
          Play with keyboard
        </button>
        <button type="button" className="clay-btn ghost" onClick={onBack}>
          Back
        </button>
      </div>
    </section>
  );
}

/** Shown while the run is frozen because no hand is visible. */
export function HandLostOverlay() {
  return (
    <section className="screen paused" role="status" aria-live="assertive">
      <div className="clay card" style={{ '--c1': '#4b4390', '--c2': '#2b2760' } as CSSProperties}>
        <p className="oops" aria-hidden="true">🖐️</p>
        <h2 className="screen-title small">Show your hand</h2>
        <p className="error-text">The run is paused. It continues when the camera sees your hand again.</p>
      </div>
    </section>
  );
}
