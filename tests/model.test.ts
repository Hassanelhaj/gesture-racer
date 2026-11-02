import { describe, expect, it } from 'vitest';
import type { GestureModel } from '../src/vision/classifier';
import { loadGestureModel } from '../src/vision/model';

const fakeOnnx: GestureModel = {
  kind: 'onnx',
  busy: false,
  predict: () => Promise.reject(new Error('unused')),
  dispose: () => Promise.resolve(),
};

describe('loadGestureModel', () => {
  it('returns the trained model when it loads', async () => {
    expect(await loadGestureModel(() => Promise.resolve(fakeOnnx))).toBe(fakeOnnx);
  });

  it('falls back to the built-in rules when the trained model is missing', async () => {
    const model = await loadGestureModel(() => Promise.reject(new Error('404 gesture.int8.onnx')));
    expect(model.kind).toBe('rules');
  });
});
