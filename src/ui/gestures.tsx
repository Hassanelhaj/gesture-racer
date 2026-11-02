import type { Gesture } from '../types';

/** Player-facing description and look of each gesture, shared by every screen. */
export interface GestureInfo {
  readonly title: string;
  readonly hint: string;
  readonly glyph: string;
  /** Rotation of the glyph in degrees, to show the tilt. */
  readonly tilt: number;
  /** CSS colour variables for the clay card. */
  readonly colors: readonly [string, string];
}

export const GESTURE_INFO: Readonly<Record<Gesture, GestureInfo>> = {
  LEFT: { title: 'Left', hint: 'Tilt your open hand left', glyph: '🖐️', tilt: -28, colors: ['#8fe3ff', '#4aa8e8'] },
  RIGHT: { title: 'Right', hint: 'Tilt your open hand right', glyph: '🖐️', tilt: 28, colors: ['#b9a6ff', '#7d62e8'] },
  JUMP: { title: 'Jump', hint: 'Close your hand into a fist', glyph: '✊', tilt: 0, colors: ['#ffd78a', '#f0a23a'] },
};

interface GlyphProps {
  readonly gesture: Gesture;
}

/** The emoji hand, rotated to match the gesture. Decorative: the label carries the meaning. */
export function GestureGlyph({ gesture }: GlyphProps) {
  const info = GESTURE_INFO[gesture];
  return (
    <span className="glyph" style={{ transform: `rotate(${info.tilt}deg)` }} aria-hidden="true">
      {info.glyph}
    </span>
  );
}
