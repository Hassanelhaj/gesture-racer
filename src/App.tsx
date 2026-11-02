import { useEffect, useRef, useState, useSyncExternalStore } from 'react';
import type { PointerEvent } from 'react';
import { GameSession } from './session';
import { GameOver } from './ui/GameOver';
import { HUD } from './ui/HUD';
import { Settings } from './ui/Settings';
import { PracticeScreen, StartScreen } from './ui/StartScreen';
import { ErrorOverlay, HandLostOverlay, LoadingOverlay } from './ui/StatusOverlay';

interface AppProps {
  /** Injectable so demos and tests can supply a session with a scripted vision source. */
  readonly session?: GameSession;
}

/**
 * Root component. All game state lives in GameSession; this file only turns
 * the current snapshot into screens and forwards clicks back as actions.
 */
export default function App({ session: injected }: AppProps = {}) {
  const [session] = useState(() => injected ?? new GameSession());
  const snap = useSyncExternalStore(session.subscribe, session.getSnapshot);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (canvas === null || video === null) return undefined;
    return session.attach(canvas, video);
  }, [session]);

  const openSettings = (): void => {
    setSettingsOpen(true);
  };

  const onStagePointerDown = (event: PointerEvent<HTMLDivElement>): void => {
    const box = event.currentTarget.getBoundingClientRect();
    session.tap((event.clientX - box.left) / box.width);
  };

  const cameraMode = snap.inputMode === 'camera';
  // The video element must stay mounted: the tracker reads frames from it.
  const showBubble = cameraMode && (snap.screen === 'practice' || (snap.screen === 'playing' && snap.settings.cameraPreview));
  const bubbleClass = `cam-bubble${snap.screen === 'practice' ? ' large' : ''}${showBubble ? '' : ' hidden'}${snap.vision?.handPresent === true ? ' tracking' : ''}`;

  return (
    <main className="app">
      <div className="blob blob-a" aria-hidden="true" />
      <div className="blob blob-b" aria-hidden="true" />

      <div className="stage-frame">
        <div className="stage" onPointerDown={onStagePointerDown}>
          <canvas ref={canvasRef} className="game-canvas" role="img" aria-label="Racing game view" />
          <div className="ui">
            <video ref={videoRef} className={bubbleClass} playsInline muted aria-hidden="true" />

            {snap.screen === 'start' && (
              <StartScreen best={snap.best} onCamera={() => void session.startCamera()} onKeyboard={() => { session.startKeyboard(); }} onSettings={openSettings} />
            )}
            {snap.screen === 'loading' && <LoadingOverlay message={snap.loadingMessage} fraction={snap.loadingFraction} />}
            {snap.screen === 'practice' && (
              <PracticeScreen status={snap.vision} done={snap.practiceDone} onStart={() => { session.startRun(); }} onSettings={openSettings} />
            )}
            {snap.screen === 'playing' && (
              <>
                <HUD hud={snap.hud} best={snap.best} inputMode={snap.inputMode} vision={snap.vision} />
                {snap.handPaused && <HandLostOverlay />}
              </>
            )}
            {snap.screen === 'gameover' && snap.result !== null && (
              <GameOver
                result={snap.result}
                best={snap.best}
                isNewBest={snap.newBest}
                onRetry={() => { session.startRun(); }}
                onMenu={() => void session.toMenu()}
                onSettings={openSettings}
              />
            )}
            {snap.screen === 'error' && snap.error !== null && (
              <ErrorOverlay
                message={snap.error.message}
                canRetry={cameraMode}
                onRetry={() => void session.startCamera()}
                onKeyboard={() => { session.startKeyboard(); }}
                onBack={() => void session.toMenu()}
              />
            )}

            {settingsOpen && (
              <Settings
                settings={snap.settings}
                vision={snap.vision}
                perf={snap.perf}
                onChange={(patch) => { session.setSettings(patch); }}
                onClose={() => { setSettingsOpen(false); }}
              />
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
