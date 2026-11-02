import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GestureClassifier } from '../src/vision/classifier';

const ort = vi.hoisted(() => ({
  loads: 0,
  create: vi.fn(),
}));

// A fake ONNX Runtime. The factory only runs when the module is actually imported,
// which lets the tests prove that a missing model never loads the 15 MB runtime.
vi.mock('onnxruntime-web/wasm', () => {
  ort.loads += 1;
  return {
    env: { wasm: {} },
    InferenceSession: { create: ort.create },
    Tensor: class {
      constructor(
        readonly type: string,
        readonly data: Float32Array,
        readonly dims: number[],
      ) {}
    },
  };
});

const okSession = () => ({
  inputNames: ['features'],
  outputNames: ['logits'],
  run: () => Promise.resolve({ logits: { data: new Float32Array([6, 0, 0]) } }),
  release: () => Promise.resolve(),
});

function stubModelResponse(response: Response): void {
  vi.stubGlobal('fetch', () => Promise.resolve(response));
}

beforeEach(() => {
  ort.create.mockReset();
  ort.create.mockImplementation(() => Promise.resolve(okSession()));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('GestureClassifier.create without a usable model file', () => {
  it.each([
    ['a 404', () => new Response('not found', { status: 404 })],
    ['an HTML page served with 200', () => new Response('<!doctype html>', { status: 200, headers: { 'content-type': 'text/html' } })],
    ['an empty file', () => new Response(new Uint8Array(0), { status: 200 })],
  ])('rejects %s without ever loading ONNX Runtime', async (_label, makeResponse) => {
    stubModelResponse(makeResponse());
    await expect(GestureClassifier.create('/models/x.onnx')).rejects.toMatchObject({ code: 'classifier-load-failed' });
    expect(ort.loads).toBe(0);
    expect(ort.create).not.toHaveBeenCalled();
  });

  it('rejects when the network is down', async () => {
    vi.stubGlobal('fetch', () => Promise.reject(new TypeError('Failed to fetch')));
    await expect(GestureClassifier.create('/models/x.onnx')).rejects.toMatchObject({ code: 'classifier-load-failed' });
    expect(ort.loads).toBe(0);
  });
});

describe('GestureClassifier.create with a model file', () => {
  it('loads the runtime lazily, creates the session from the downloaded bytes, and classifies', async () => {
    stubModelResponse(new Response(new Uint8Array([1, 2, 3, 4]), { status: 200, headers: { 'content-type': 'application/octet-stream' } }));
    const classifier = await GestureClassifier.create('/models/x.onnx');
    expect(ort.loads).toBe(1);
    const firstArg: unknown = ort.create.mock.calls[0]?.[0];
    expect(firstArg).toBeInstanceOf(Uint8Array);
    expect(Array.from(firstArg as Uint8Array)).toEqual([1, 2, 3, 4]);

    const prediction = await classifier.predict(new Float32Array(12));
    expect(prediction.gesture).toBe('LEFT');
    expect(classifier.kind).toBe('onnx');
  });

  it('wraps a runtime failure (corrupt model) as classifier-load-failed', async () => {
    stubModelResponse(new Response(new Uint8Array([9, 9, 9]), { status: 200 }));
    ort.create.mockImplementation(() => Promise.reject(new Error('invalid protobuf')));
    await expect(GestureClassifier.create('/models/x.onnx')).rejects.toMatchObject({ code: 'classifier-load-failed' });
  });

  it('rejects a model whose output does not match the 3 gestures', async () => {
    stubModelResponse(new Response(new Uint8Array([1]), { status: 200 }));
    ort.create.mockImplementation(() =>
      Promise.resolve({ ...okSession(), run: () => Promise.resolve({ logits: { data: new Float32Array([1, 2]) } }) }),
    );
    await expect(GestureClassifier.create('/models/x.onnx')).rejects.toMatchObject({ code: 'classifier-load-failed' });
  });
});
