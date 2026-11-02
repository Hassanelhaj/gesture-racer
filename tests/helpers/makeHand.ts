import type { HandLandmarks, Landmark } from '../../src/types';
import { FEATURES } from '../../src/constants';

/** Options for the synthetic hand used throughout the vision tests. */
export interface HandSpec {
  /** In-plane tilt in radians; positive = tilted to the player's right (selfie view). */
  readonly roll?: number;
  /** Curl of each finger, 0 = straight, 1 = fully folded (same scale as the curl features). */
  readonly curl?: number;
  /** Palm length (wrist to middle knuckle) in image-height units. */
  readonly palmLength?: number;
  /** Wrist position in image-height units, selfie (mirrored) view, y down. */
  readonly wrist?: { readonly x: number; readonly y: number };
  /** Video aspect ratio used to normalise x. */
  readonly aspect?: number;
}

type V2 = readonly [number, number];

const rotate = ([x, y]: V2, angle: number): V2 => [
  x * Math.cos(angle) - y * Math.sin(angle),
  x * Math.sin(angle) + y * Math.cos(angle),
];

/**
 * Builds 21 MediaPipe-ordered landmarks for a flat (z = 0) synthetic hand.
 *
 * Fingers are generated from joint angles, so a requested curl of c produces a
 * PIP-joint angle that the feature extractor must map back to ~c. That makes
 * the expected value of every feature known in closed form.
 */
export function makeHand(spec: HandSpec = {}): HandLandmarks {
  const roll = spec.roll ?? 0;
  const curl = spec.curl ?? 0;
  const palmLength = spec.palmLength ?? 0.2;
  const wrist = spec.wrist ?? { x: 0.8, y: 0.8 };
  const aspect = spec.aspect ?? 1;

  // Local frame: palm length = 1, +y points from wrist toward the fingers.
  const local = new Map<number, V2>();
  local.set(0, [0, 0]);

  const place = (mcpIndex: number, mcp: V2, dir: V2, foldedAngle: number, segment: number): void => {
    const phi = Math.PI - curl * (Math.PI - foldedAngle); // interior angle at the PIP joint
    const pip: V2 = [mcp[0] + dir[0] * segment, mcp[1] + dir[1] * segment];
    const back: V2 = [-dir[0], -dir[1]];
    const toTip = rotate(back, phi);
    const tip: V2 = [pip[0] + toTip[0] * segment, pip[1] + toTip[1] * segment];
    const dip: V2 = [(pip[0] + tip[0]) / 2, (pip[1] + tip[1]) / 2];
    local.set(mcpIndex, mcp);
    local.set(mcpIndex + 1, pip);
    local.set(mcpIndex + 2, dip);
    local.set(mcpIndex + 3, tip);
  };

  // Thumb occupies 1 (CMC), 2 (MCP), 3 (IP), 4 (TIP); the MCP-IP-TIP angle is what we curl.
  local.set(1, [-0.3, 0.3]);
  const thumbDir: V2 = [-0.5, 0.85];
  const thumbNorm = Math.hypot(thumbDir[0], thumbDir[1]);
  const thumbUnit: V2 = [thumbDir[0] / thumbNorm, thumbDir[1] / thumbNorm];
  {
    const mcp: V2 = [-0.45, 0.5];
    const phi = Math.PI - curl * (Math.PI - FEATURES.thumbFoldedAngleRad);
    const ip: V2 = [mcp[0] + thumbUnit[0] * 0.3, mcp[1] + thumbUnit[1] * 0.3];
    const toTip = rotate([-thumbUnit[0], -thumbUnit[1]], phi);
    const tip: V2 = [ip[0] + toTip[0] * 0.3, ip[1] + toTip[1] * 0.3];
    local.set(2, mcp);
    local.set(3, ip);
    local.set(4, tip);
  }

  const fingerSegment = 0.4;
  place(5, [-0.25, 0.95], [-0.08, 1], FEATURES.fingerFoldedAngleRad, fingerSegment); // index
  place(9, [0, 1], [0, 1], FEATURES.fingerFoldedAngleRad, fingerSegment); // middle
  place(13, [0.22, 0.95], [0.06, 1], FEATURES.fingerFoldedAngleRad, fingerSegment); // ring
  place(17, [0.42, 0.85], [0.14, 1], FEATURES.fingerFoldedAngleRad, fingerSegment); // pinky

  const landmarks: Landmark[] = [];
  for (let i = 0; i < 21; i += 1) {
    const [lx, ly] = local.get(i) ?? [0, 0];
    // Clockwise rotation by `roll` (y up), so roll > 0 tilts the fingers to the right.
    const rx = lx * Math.cos(roll) + ly * Math.sin(roll);
    const ry = -lx * Math.sin(roll) + ly * Math.cos(roll);
    const selfieX = wrist.x + rx * palmLength; // image-height units
    const y = wrist.y - ry * palmLength; // screen y grows downward
    // Normalise to MediaPipe space, then un-mirror: raw camera x = 1 - selfie x.
    landmarks.push({ x: 1 - selfieX / aspect, y, z: 0 });
  }
  return landmarks;
}
