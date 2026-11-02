import { AUDIO } from '../constants';
import { createLogger } from '../logger';

const log = createLogger('game/audio');

type AudioContextCtor = typeof AudioContext;

function getAudioContextCtor(): AudioContextCtor | null {
  const w = window as unknown as { AudioContext?: AudioContextCtor; webkitAudioContext?: AudioContextCtor };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

const lerp = (from: number, to: number, t: number): number => from + (to - from) * t;

/**
 * Procedural sound: no audio files, so nothing to download and nothing to license.
 *
 * - engine : a sawtooth through a low-pass filter; pitch and brightness rise with speed
 * - jump   : a short rising sine "boing"
 * - crash  : filtered noise burst plus a falling thump
 * - lane   : a very quiet tick
 *
 * Browsers only allow audio after a user gesture, so `start()` must be called
 * from a click/key handler (the Start button). Every method is a safe no-op
 * when audio is unavailable or muted: sound is never required to play.
 */
export class AudioEngine {
  private context: AudioContext | null = null;
  private master: GainNode | null = null;
  private engineOsc: OscillatorNode | null = null;
  private engineFilter: BiquadFilterNode | null = null;
  private engineGain: GainNode | null = null;
  private noiseBuffer: AudioBuffer | null = null;
  private muted = false;

  /**
   * Creates the audio graph and starts the engine hum.
   * @returns false if Web Audio is unsupported or could not start
   */
  async start(): Promise<boolean> {
    if (this.context !== null) {
      await this.context.resume();
      return true;
    }
    const Ctor = getAudioContextCtor();
    if (Ctor === null) {
      log.warn('Web Audio not available; continuing without sound');
      return false;
    }
    try {
      const context = new Ctor();
      await context.resume();

      const master = context.createGain();
      master.gain.value = this.muted ? 0 : AUDIO.masterGain;
      master.connect(context.destination);

      const engineGain = context.createGain();
      engineGain.gain.value = AUDIO.engineGain;
      const engineFilter = context.createBiquadFilter();
      engineFilter.type = 'lowpass';
      engineFilter.frequency.value = AUDIO.engineFilterMinHz;
      const engineOsc = context.createOscillator();
      engineOsc.type = 'sawtooth';
      engineOsc.frequency.value = AUDIO.engineMinHz;
      engineOsc.connect(engineFilter).connect(engineGain).connect(master);
      engineOsc.start();

      // One reusable noise buffer for every crash.
      const length = Math.floor(context.sampleRate * AUDIO.crashNoiseS);
      const noise = context.createBuffer(1, length, context.sampleRate);
      const data = noise.getChannelData(0);
      for (let i = 0; i < length; i += 1) data[i] = Math.random() * 2 - 1;

      this.context = context;
      this.master = master;
      this.engineGain = engineGain;
      this.engineFilter = engineFilter;
      this.engineOsc = engineOsc;
      this.noiseBuffer = noise;
      return true;
    } catch (error) {
      log.warn('Audio failed to start; continuing without sound', error);
      return false;
    }
  }

  /** Mutes or unmutes everything with a short ramp (no click). */
  setMuted(muted: boolean): void {
    this.muted = muted;
    if (this.context !== null && this.master !== null) {
      this.master.gain.setTargetAtTime(muted ? 0 : AUDIO.masterGain, this.context.currentTime, 0.02);
    }
  }

  /** Sets engine pitch/brightness from game difficulty (0..1). */
  setEngine(difficulty: number): void {
    if (this.context === null || this.engineOsc === null || this.engineFilter === null) return;
    const t = this.context.currentTime;
    this.engineOsc.frequency.setTargetAtTime(
      lerp(AUDIO.engineMinHz, AUDIO.engineMaxHz, difficulty), t, 0.1);
    this.engineFilter.frequency.setTargetAtTime(
      lerp(AUDIO.engineFilterMinHz, AUDIO.engineFilterMaxHz, difficulty), t, 0.1);
  }

  /** Fades the engine in or out (used when a run ends or restarts). */
  setEngineOn(on: boolean): void {
    if (this.context === null || this.engineGain === null) return;
    this.engineGain.gain.setTargetAtTime(on ? AUDIO.engineGain : 0, this.context.currentTime, 0.08);
  }

  playJump(): void {
    this.blip('sine', 280, 620, 0.18, 0.5);
  }

  playLane(): void {
    this.blip('triangle', 200, 160, 0.05, 0.12);
  }

  playPass(): void {
    this.blip('sine', 660, 880, 0.08, 0.15);
  }

  playCrash(): void {
    const context = this.context;
    const master = this.master;
    if (context === null || master === null || this.noiseBuffer === null) return;
    const now = context.currentTime;
    this.setEngineOn(false);

    const noise = context.createBufferSource();
    noise.buffer = this.noiseBuffer;
    const band = context.createBiquadFilter();
    band.type = 'lowpass';
    band.frequency.setValueAtTime(3000, now);
    band.frequency.exponentialRampToValueAtTime(200, now + AUDIO.crashNoiseS);
    const noiseGain = context.createGain();
    noiseGain.gain.setValueAtTime(1, now);
    noiseGain.gain.exponentialRampToValueAtTime(0.001, now + AUDIO.crashNoiseS);
    noise.connect(band).connect(noiseGain).connect(master);
    noise.start(now);

    this.blip('sine', 140, 40, 0.4, 1);
  }

  /** Releases the audio graph. The engine can be restarted with `start()`. */
  async stop(): Promise<void> {
    this.engineOsc?.stop();
    const context = this.context;
    this.context = null;
    this.master = null;
    this.engineGain = null;
    this.engineFilter = null;
    this.engineOsc = null;
    this.noiseBuffer = null;
    await context?.close();
  }

  /** A single enveloped oscillator sweep. */
  private blip(type: OscillatorType, fromHz: number, toHz: number, seconds: number, peak: number): void {
    const context = this.context;
    const master = this.master;
    if (context === null || master === null) return;
    const now = context.currentTime;
    const osc = context.createOscillator();
    osc.type = type;
    osc.frequency.setValueAtTime(fromHz, now);
    osc.frequency.exponentialRampToValueAtTime(toHz, now + seconds);
    const gain = context.createGain();
    gain.gain.setValueAtTime(peak, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + seconds);
    osc.connect(gain).connect(master);
    osc.start(now);
    osc.stop(now + seconds + 0.02);
  }
}
