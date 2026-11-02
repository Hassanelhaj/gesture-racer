import { FEATURES } from '../constants';
import type { FeatureVector, HandLandmarks, Landmark } from '../types';

/** Number of landmarks MediaPipe returns for one hand. */
export const LANDMARK_COUNT = 21;

/**
 * Feature order. ORDER IS A CONTRACT shared with ml/train.py and the data
 * collector: the classifier input at index i is the feature named here at i.
 */
export const FEATURE_NAMES = [
  'roll_sin',
  'roll_cos',
  'wrist_x',
  'wrist_y',
  'palm_scale',
  'curl_thumb',
  'curl_index',
  'curl_middle',
  'curl_ring',
  'curl_pinky',
  'finger_spread',
  'hand_openness',
] as const;

/** MediaPipe hand landmark indices used below. */
const LM = {
  WRIST: 0,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_PIP: 14,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_TIP: 20,
} as const;

const FINGERTIPS = [LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP] as const;

interface Vec3 {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const length = (a: Vec3): number => Math.sqrt(dot(a, a));
const distance = (a: Vec3, b: Vec3): number => length(sub(a, b));

/**
 * Interior angle at `joint` between the directions to `a` and `b`, in radians
 * (0 = folded back on itself, PI = perfectly straight).
 * A zero-length arm returns PI ("straight"), the neutral value, so noise on a
 * collapsed joint never reads as a curled finger.
 */
function jointAngle(a: Vec3, joint: Vec3, b: Vec3): number {
  const u = sub(a, joint);
  const v = sub(b, joint);
  const denominator = length(u) * length(v);
  if (denominator < 1e-9) return Math.PI;
  return Math.acos(clamp(dot(u, v) / denominator, -1, 1));
}

/**
 * Maps a joint angle to a 0..1 curl: 0 = straight finger, 1 = folded to or
 * past `foldedAngle`. Angle-based (not distance-based) so it is invariant to
 * hand size and distance from the camera.
 */
function curl(mcp: Vec3, middle: Vec3, tip: Vec3, foldedAngle: number): number {
  const angle = jointAngle(mcp, middle, tip);
  return clamp((Math.PI - angle) / (Math.PI - foldedAngle), 0, 1);
}

/**
 * Builds the 12-dim classifier input from one hand.
 *
 * WHY these features: LEFT/RIGHT are "open palm tilted" and JUMP is "fist", so
 * what matters is in-plane hand rotation (roll) and how curled the fingers
 * are. Everything is expressed relative to the hand itself (angles, ratios) so
 * the model generalises across hand sizes and camera distances. Raw landmark
 * coordinates are deliberately NOT fed to the model.
 *
 * COORDINATE CHOICES (all assumptions, validated by tests/features.test.ts):
 *  - x is mirrored (1 - x) so "left" means left in the selfie-view the player
 *    sees on screen, not left in the raw camera image.
 *  - x and z are multiplied by the video aspect ratio so all axes share one
 *    unit (image height). Without this, angles distort on 16:9 vs 4:3 cameras.
 *
 * Output (see FEATURE_NAMES), all bounded to [-1, 1] or [0, 1]:
 *   0,1  roll_sin, roll_cos : tilt of the wrist->middle-knuckle axis from
 *                             vertical; sin > 0 = tilted to the player's right.
 *                             (sin, cos) avoids the +-PI wrap-around of a raw angle.
 *   2,3  wrist_x, wrist_y   : wrist position in the mirrored frame, 0..1.
 *   4    palm_scale         : palm length / palmLengthRef, a distance proxy.
 *   5-9  curl_*             : thumb, index, middle, ring, pinky; 0 straight, 1 folded.
 *   10   finger_spread      : index-tip to pinky-tip distance over knuckle width.
 *   11   hand_openness      : mean fingertip-to-wrist distance over palm length.
 *
 * @param landmarks   exactly 21 MediaPipe landmarks of one hand
 * @param aspectRatio video width / height (defaults to FEATURES.defaultAspectRatio)
 * @returns the vector, or null if the hand geometry is degenerate (for
 *          example the palm points straight at the camera so roll is undefined).
 *          Callers treat null like "no gesture".
 * @throws RangeError for a wrong landmark count or invalid aspect ratio, which
 *         are programming errors rather than runtime conditions.
 */
export function extractFeatures(
  landmarks: HandLandmarks,
  aspectRatio: number = FEATURES.defaultAspectRatio,
): FeatureVector | null {
  if (landmarks.length !== LANDMARK_COUNT) {
    throw new RangeError(`Expected ${LANDMARK_COUNT} landmarks, got ${landmarks.length}`);
  }
  if (!Number.isFinite(aspectRatio) || aspectRatio <= 0) {
    throw new RangeError(`Invalid aspect ratio: ${aspectRatio}`);
  }

  const toWorking = (lm: Landmark): Vec3 => ({
    x: (1 - lm.x) * aspectRatio,
    y: lm.y,
    z: lm.z * aspectRatio,
  });
  const points = landmarks.map(toWorking);
  const at = (index: number): Vec3 => {
    const point = points[index];
    if (point === undefined) throw new RangeError(`Missing landmark ${index}`);
    return point;
  };
  const rawWrist = landmarks[LM.WRIST];
  if (rawWrist === undefined) throw new RangeError('Missing wrist landmark');

  const wrist = at(LM.WRIST);
  const palmAxis = sub(at(LM.MIDDLE_MCP), wrist);
  const palmLength2d = Math.hypot(palmAxis.x, palmAxis.y);
  const palmLength3d = length(palmAxis);
  const knuckleWidth = distance(at(LM.INDEX_MCP), at(LM.PINKY_MCP));
  if (palmLength2d < FEATURES.minGeometry || knuckleWidth < FEATURES.minGeometry) return null;

  // Image y grows downward, so an upright hand has palmAxis.y < 0.
  const rollSin = palmAxis.x / palmLength2d;
  const rollCos = -palmAxis.y / palmLength2d;

  const palmScale =
    clamp(palmLength2d / FEATURES.palmLengthRef, 0, FEATURES.palmScaleMax) / FEATURES.palmScaleMax;

  const fingerCurl = (mcp: number, pip: number, tip: number): number =>
    curl(at(mcp), at(pip), at(tip), FEATURES.fingerFoldedAngleRad);

  const spread =
    clamp(distance(at(LM.INDEX_TIP), at(LM.PINKY_TIP)) / knuckleWidth, 0, FEATURES.spreadMax) /
    FEATURES.spreadMax;

  let tipDistanceSum = 0;
  for (const tip of FINGERTIPS) tipDistanceSum += distance(at(tip), wrist);
  const openness =
    clamp(tipDistanceSum / FINGERTIPS.length / palmLength3d, 0, FEATURES.opennessMax) /
    FEATURES.opennessMax;

  const vector = new Float32Array(FEATURES.dim);
  vector[0] = rollSin;
  vector[1] = rollCos;
  vector[2] = clamp(1 - rawWrist.x, 0, 1);
  vector[3] = clamp(rawWrist.y, 0, 1);
  vector[4] = palmScale;
  vector[5] = curl(at(LM.THUMB_MCP), at(LM.THUMB_IP), at(LM.THUMB_TIP), FEATURES.thumbFoldedAngleRad);
  vector[6] = fingerCurl(LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_TIP);
  vector[7] = fingerCurl(LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_TIP);
  vector[8] = fingerCurl(LM.RING_MCP, LM.RING_PIP, LM.RING_TIP);
  vector[9] = fingerCurl(LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_TIP);
  vector[10] = spread;
  vector[11] = openness;
  return vector;
}
