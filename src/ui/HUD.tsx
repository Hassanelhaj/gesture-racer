import type { CSSProperties } from 'react';
import type { HudState } from '../game/game';
import type { InputMode } from '../session';
import type { PipelineStatus } from '../vision/pipeline';
import { GESTURE_INFO, GestureGlyph } from './gestures';

interface HudProps {
  readonly hud: HudState;
  readonly best: number;
  readonly inputMode: InputMode | null;
  readonly vision: PipelineStatus | null;
}

/** In-run overlay: score, best, speed and (camera mode) what the tracker currently sees. */
export function HUD({ hud, best, inputMode, vision }: HudProps) {
  const gesture = vision?.stableGesture ?? null;
  const tracking = vision?.handPresent === true;
  return (
    <div className="hud" aria-label="Game status">
      <div className="hud-top">
        <div className="clay stat score" style={{ '--c1': '#ff8fa8', '--c2': '#e5476b' } as CSSProperties}>
          <small>Score</small>
          <strong aria-live="off">{hud.score.toLocaleString()}</strong>
        </div>
        <div className="clay stat compact" style={{ '--c1': '#9be8cf', '--c2': '#3fbf98' } as CSSProperties}>
          <small>Best</small>
          <strong>{Math.max(best, hud.score).toLocaleString()}</strong>
        </div>
      </div>

      <div className="hud-bottom">
        <div className="clay stat compact speed" style={{ '--c1': '#b9a6ff', '--c2': '#7d62e8' } as CSSProperties}>
          <small>km/h</small>
          <strong>{hud.speedKmh}</strong>
        </div>
        {inputMode === 'camera' && (
          <div className={`clay chip${tracking ? ' on' : ' off'}`} role="status">
            {gesture !== null ? (
              <>
                <GestureGlyph gesture={gesture} />
                <span>{GESTURE_INFO[gesture].title}</span>
              </>
            ) : (
              <span>{tracking ? 'Hand tracked' : 'No hand'}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
