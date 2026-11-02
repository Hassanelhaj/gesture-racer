import { useEffect, useRef } from 'react';
import type { Settings as SettingsValues } from '../storage/settings';
import type { FramePerf } from '../session';
import type { PipelineStatus } from '../vision/pipeline';

interface SettingsProps {
  readonly settings: SettingsValues;
  readonly vision: PipelineStatus | null;
  readonly perf: FramePerf;
  readonly onChange: (patch: Partial<SettingsValues>) => void;
  readonly onClose: () => void;
}

interface ToggleProps {
  readonly label: string;
  readonly description: string;
  readonly checked: boolean;
  readonly onChange: (checked: boolean) => void;
}

/** A clay "switch": a real checkbox underneath, so keyboard and screen readers just work. */
function Toggle({ label, description, checked, onChange }: ToggleProps) {
  return (
    <label className="toggle">
      <span className="toggle-text">
        <strong>{label}</strong>
        <small>{description}</small>
      </span>
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        onChange={(event) => {
          onChange(event.target.checked);
        }}
      />
      <span className="switch" aria-hidden="true" />
    </label>
  );
}

/** Modal settings sheet. Escape or the Done button closes it. */
export function Settings({ settings, vision, perf, onChange, onClose }: SettingsProps) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
    };
  }, [onClose]);

  return (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="clay card settings"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        onClick={(event) => {
          event.stopPropagation();
        }}
      >
        <h2 id="settings-title" className="screen-title">Settings</h2>

        <Toggle label="Sound" description="Engine, jumps and crashes" checked={settings.sound} onChange={(sound) => { onChange({ sound }); }} />
        <Toggle label="Camera preview" description="Small mirrored view while driving" checked={settings.cameraPreview} onChange={(cameraPreview) => { onChange({ cameraPreview }); }} />
        <Toggle label="Preload models" description="Load the camera models in the background (about 22 MB)" checked={settings.preload} onChange={(preload) => { onChange({ preload }); }} />
        <Toggle label="Diagnostics" description="Show latency and frame rate" checked={settings.diagnostics} onChange={(diagnostics) => { onChange({ diagnostics }); }} />

        {settings.diagnostics && (
          <dl className="diagnostics" aria-label="Diagnostics">
            <div><dt>Game</dt><dd>{perf.fps.toFixed(0)} fps</dd></div>
            <div><dt>Slowest 5%</dt><dd>{perf.p95FrameMs.toFixed(0)} ms</dd></div>
            {vision === null ? (
              <p>Start the camera to see tracking numbers.</p>
            ) : (
              <>
                <div><dt>Hand model</dt><dd>{vision.delegate}</dd></div>
                <div><dt>Classifier</dt><dd>{vision.classifier === 'onnx' ? 'ONNX model' : 'Built-in rules'}</dd></div>
                <div><dt>Camera</dt><dd>{vision.cameraFps.toFixed(0)} fps</dd></div>
                <div><dt>Tracking</dt><dd>{vision.throttled ? 'every 2nd frame' : 'every frame'}</dd></div>
                <div><dt>Detect</dt><dd>{vision.detectMs.toFixed(1)} ms</dd></div>
                <div><dt>Classify</dt><dd>{vision.inferenceMs.toFixed(2)} ms</dd></div>
                <div><dt>Confidence</dt><dd>{(vision.confidence * 100).toFixed(0)}%</dd></div>
              </>
            )}
          </dl>
        )}

        <div className="actions">
          <button ref={closeRef} type="button" className="clay-btn primary" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
