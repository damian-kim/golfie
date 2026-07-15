import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "../lib/api";
import "./SwingReplayModal.css";

interface SwingReplayModalProps {
  sessionId: string;
  onClose: () => void;
  onPlayBallFlight: () => void;
  onVideoError: () => void;
  error?: string | null;
}

const SPEEDS = [0.25, 0.5, 1, 1.5, 2];

export function SwingReplayModal({ sessionId, onClose, onPlayBallFlight, onVideoError, error }: SwingReplayModalProps) {
  const videos = useRef<HTMLVideoElement[]>([]);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [frameRate, setFrameRate] = useState(30);

  const registerVideo = (element: HTMLVideoElement | null) => {
    if (element && !videos.current.includes(element)) videos.current.push(element);
  };

  const seekAll = useCallback((time: number) => {
    const bounded = Math.max(0, Math.min(time, duration || Number.POSITIVE_INFINITY));
    videos.current.forEach((video) => {
      if (Number.isFinite(video.duration)) video.currentTime = Math.min(bounded, video.duration);
    });
    setCurrentTime(bounded);
  }, [duration]);

  const setAllPlaying = useCallback((shouldPlay: boolean) => {
    setPlaying(shouldPlay);
    videos.current.forEach((video) => {
      video.playbackRate = speed;
      if (shouldPlay) void video.play().catch(() => setPlaying(false));
      else video.pause();
    });
  }, [speed]);

  useEffect(() => {
    fetch(`${API_BASE_URL}/sessions/${sessionId}/video/camera_a/frame_map`)
      .then((response) => response.ok ? response.json() : null)
      .then((mapping) => {
        if (mapping?.fps_used > 0) setFrameRate(mapping.fps_used);
      })
      .catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    videos.current.forEach((video) => { video.playbackRate = speed; });
  }, [speed]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        setAllPlaying(false);
        seekAll(currentTime + (event.key === "ArrowRight" ? 1 : -1) / frameRate);
      } else if (event.key === " ") {
        event.preventDefault();
        setAllPlaying(!playing);
      } else if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [currentTime, frameRate, onClose, playing, seekAll, setAllPlaying]);

  const handleMasterTime = (event: React.SyntheticEvent<HTMLVideoElement>) => {
    const master = event.currentTarget;
    setCurrentTime(master.currentTime);
    videos.current.forEach((video) => {
      if (video !== master && Math.abs(video.currentTime - master.currentTime) > 0.06) {
        video.currentTime = Math.min(master.currentTime, Number.isFinite(video.duration) ? video.duration : master.currentTime);
      }
    });
  };

  const videoUrl = (camera: "camera_a" | "camera_b", kind: "stripped" | "replay-original") =>
    `${API_BASE_URL}/sessions/${sessionId}/video/${camera}/${kind}`;

  return (
    <div className="swing-replay" role="dialog" aria-modal="true" aria-label="Swing comparison replay">
      <div className="swing-replay__panel">
        <header className="swing-replay__header">
          <div>
            <h2>Swing comparison</h2>
            <p>Body contour and pose guides above · original uploaded frames below</p>
          </div>
          <button onClick={onClose} aria-label="Close swing replay">✕</button>
        </header>

        <div className="swing-replay__legend" aria-label="Pose overlay legend">
          <span><i className="swing-replay__key swing-replay__key--left" />Anatomical left arm</span>
          <span><i className="swing-replay__key swing-replay__key--right" />Anatomical right arm</span>
          <span><i className="swing-replay__key swing-replay__key--shoulders" />Shoulder axis</span>
          <span><i className="swing-replay__key swing-replay__key--hips" />Hip axis</span>
          <span><i className="swing-replay__key swing-replay__key--balance" />2D posture center</span>
          <small>Left/right may appear mirrored by the source video. Posture center is not a pressure measurement.</small>
        </div>

        <div className="swing-replay__cameras">
          {(["camera_a", "camera_b"] as const).map((camera, cameraIndex) => (
            <section className="swing-replay__camera" key={camera}>
              <h3>{camera === "camera_a" ? "Camera A · Down the line" : "Camera B · Face on"}</h3>
              <figure>
                <figcaption>Body + pose analysis</figcaption>
                <video
                  ref={registerVideo}
                  src={videoUrl(camera, "stripped")}
                  muted playsInline autoPlay
                  onLoadedMetadata={(event) => {
                    event.currentTarget.playbackRate = speed;
                    if (cameraIndex === 0) setDuration(event.currentTarget.duration);
                  }}
                  onTimeUpdate={cameraIndex === 0 ? handleMasterTime : undefined}
                  onEnded={cameraIndex === 0 ? () => setAllPlaying(false) : undefined}
                  onError={onVideoError}
                />
              </figure>
              <figure>
                <figcaption>Original video</figcaption>
                <video ref={registerVideo} src={videoUrl(camera, "replay-original")} muted playsInline autoPlay onError={onVideoError} />
              </figure>
            </section>
          ))}
        </div>

        {error && <div className="swing-replay__error">{error}</div>}

        <div className="swing-replay__timeline">
          <input
            aria-label="Replay position"
            type="range" min={0} max={duration || 1} step={1 / frameRate}
            value={Math.min(currentTime, duration || 1)}
            onChange={(event) => { setAllPlaying(false); seekAll(Number(event.target.value)); }}
          />
          <span>{currentTime.toFixed(2)}s / {duration.toFixed(2)}s</span>
        </div>

        <footer className="swing-replay__controls">
          <div className="swing-replay__transport">
            <button onClick={() => { setAllPlaying(false); seekAll(currentTime - 1 / frameRate); }} title="Previous frame (Left arrow)">│◀</button>
            <button className="swing-replay__play" onClick={() => setAllPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
            <button onClick={() => { setAllPlaying(false); seekAll(currentTime + 1 / frameRate); }} title="Next frame (Right arrow)">▶│</button>
          </div>
          <label>
            Speed
            <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
              {SPEEDS.map((value) => <option value={value} key={value}>{value}×</option>)}
            </select>
          </label>
          <span className="swing-replay__hint">← → frame step · Space play/pause</span>
          <button className="swing-replay__flight" onClick={onPlayBallFlight}>Play ball flight →</button>
        </footer>
      </div>
    </div>
  );
}
