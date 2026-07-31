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

type Camera = "camera_a" | "camera_b";
type CoachStatus = "priority" | "refine" | "strength";

interface CoachFinding {
  phase: string;
  time: number;
  title: string;
  status: CoachStatus;
  summary: string;
  impact: string;
  fix: string;
  drill: string;
}

const SPEEDS = [0.25, 0.5, 1, 1.5, 2];
const SWING_SCORE = 76;
const FINDINGS: CoachFinding[] = [
  { phase: "Address", time: 0.2, title: "Pressure begins slightly trail-side", status: "refine", summary: "Your posture center starts behind the midpoint of the stance, creating a small lateral recovery before rotation begins.", impact: "A centered athletic start makes the takeaway rotational and improves low-point consistency.", fix: "Set pressure beneath both shoelaces and move the lead hip one inch toward the target without changing spine tilt.", drill: "Hold address for three breaths, then make ten waist-high rehearsals while keeping your center between the ankles." },
  { phase: "Takeaway", time: 0.58, title: "Hands separate from the torso early", status: "priority", summary: "Arm travel outpaces the shoulder turn in the opening move, reducing connection and narrowing the return path.", impact: "An arm-led takeaway makes transition steeper and forces timing corrections near impact.", fix: "Move the chest, hands, and club away together. Keep the trail elbow soft until the hands pass the trail thigh.", drill: "Place a glove under the trail armpit and make ten slow takeaways to lead-arm-parallel without dropping it." },
  { phase: "Top", time: 0.92, title: "Arm width collapses near the top", status: "priority", summary: "Elbow spacing contracts while the hands continue traveling after the torso turn has nearly stopped.", impact: "Lost width steepens the first move down and makes face control depend on late hand action.", fix: "Finish the backswing when the shoulder turn finishes and feel the hands stay farther from the trail ear.", drill: "Make three-quarter pause swings. Stop for two seconds at the top, confirm width, then rotate through at half speed." },
  { phase: "Transition", time: 1.2, title: "Upper and lower body start together", status: "priority", summary: "The shoulder and hip lines begin unwinding in the same interval, leaving little pelvis-first separation.", impact: "A simultaneous transition reduces rotational speed and sends the arms outward.", fix: "Pressure the lead foot first, let the belt buckle begin opening, and allow the hands to fall before the chest releases.", drill: "Use a step-through drill: start feet together, step toward the target before the backswing completes, then swing through." },
  { phase: "Delivery", time: 1.48, title: "Hip depth decreases through delivery", status: "refine", summary: "The pelvis moves toward the ball as the legs extend, reducing room for the hands through impact.", impact: "Early extension crowds the handle and makes strike height and face delivery less predictable.", fix: "Keep the trail hip back while the lead hip clears. Rotate around the lead heel instead of pushing forward.", drill: "Set a chair lightly against your hips and make slow swings while keeping one hip in contact through impact." },
  { phase: "Finish", time: 1.88, title: "Balanced finish is a strength", status: "strength", summary: "The swing reaches a tall, supported finish with the torso stacked over the lead side.", impact: "A stable finish shows that momentum is managed through the entire motion.", fix: "Preserve this finish while changing transition. Do not add speed until you can hold it for two seconds.", drill: "Hit sets of five at 70% speed and score only whether you can hold the finish for a two-count." },
];

export function SwingReplayModal({ sessionId, onClose, onPlayBallFlight, onVideoError, error }: SwingReplayModalProps) {
  const videos = useRef(new Map<string, HTMLVideoElement>());
  const [mode, setMode] = useState<"comparison" | "coach">("comparison");
  const [findingIndex, setFindingIndex] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [frameRate, setFrameRate] = useState(30);
  const finding = FINDINGS[findingIndex];

  const registerVideo = (key: string) => (element: HTMLVideoElement | null) => {
    if (element) videos.current.set(key, element);
    else videos.current.delete(key);
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
      .then((mapping) => { if (mapping?.fps_used > 0) setFrameRate(mapping.fps_used); })
      .catch(() => {});
  }, [sessionId]);

  useEffect(() => {
    videos.current.forEach((video) => { video.playbackRate = speed; });
  }, [speed, mode]);

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
      } else if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [currentTime, frameRate, onClose, playing, seekAll, setAllPlaying]);

  const videoUrl = (camera: Camera, kind: "stripped" | "replay-original") =>
    `${API_BASE_URL}/sessions/${sessionId}/video/${camera}/${kind}`;

  const syncFromMaster = (master: HTMLVideoElement) => {
    setCurrentTime(master.currentTime);
    videos.current.forEach((video) => {
      if (video !== master && Math.abs(video.currentTime - master.currentTime) > 0.06) {
        video.currentTime = Math.min(master.currentTime, Number.isFinite(video.duration) ? video.duration : master.currentTime);
      }
    });
  };

  const chooseFinding = (index: number) => {
    setFindingIndex(index);
    setAllPlaying(false);
    seekAll(FINDINGS[index].time);
  };

  return (
    <div className="swing-replay" role="dialog" aria-modal="true" aria-label="Golfie swing analysis">
      <div className={`swing-replay__panel swing-replay__panel--${mode}`}>
        <header className="swing-replay__header">
          <div>
            <span>GOLFIE AI COACH</span>
            <h2>{mode === "comparison" ? "Swing comparison" : "Your swing, phase by phase"}</h2>
          </div>
          {mode === "comparison" && <div className="swing-score"><strong>{SWING_SCORE}</strong><span>/100<br />COMPOSITE</span></div>}
          {mode === "coach" && <button className="swing-replay__back" onClick={() => setMode("comparison")}>Back to comparison</button>}
          <button className="swing-replay__close" onClick={onClose} aria-label="Close swing replay">×</button>
        </header>

        {mode === "comparison" ? (
          <div className="swing-replay__cameras">
            {(["camera_a", "camera_b"] as const).map((camera, cameraIndex) => (
              <section className="swing-replay__camera" key={camera}>
                <h3>{camera === "camera_a" ? "Down the line" : "Face on"}</h3>
                <div className="swing-replay__pair">
                  <figure><figcaption>YOLO capture</figcaption><video ref={registerVideo(`${camera}-outline`)} src={videoUrl(camera, "stripped")} muted playsInline autoPlay onLoadedMetadata={(event) => { event.currentTarget.playbackRate = speed; if (cameraIndex === 0) setDuration(event.currentTarget.duration); }} onTimeUpdate={cameraIndex === 0 ? (event) => syncFromMaster(event.currentTarget) : undefined} onEnded={cameraIndex === 0 ? () => setAllPlaying(false) : undefined} onError={onVideoError} /></figure>
                  <figure><figcaption>Original</figcaption><video ref={registerVideo(`${camera}-original`)} src={videoUrl(camera, "replay-original")} muted playsInline autoPlay onError={onVideoError} /></figure>
                </div>
              </section>
            ))}
          </div>
        ) : (
          <div className="coach-layout">
            <nav className="coach-phases" aria-label="Swing phases">
              {FINDINGS.map((item, index) => <button key={item.phase} className={index === findingIndex ? "active" : ""} onClick={() => chooseFinding(index)}><span>0{index + 1}</span><strong>{item.phase}</strong><i className={`status-${item.status}`} /></button>)}
            </nav>
            <div className="coach-video-grid">
              {(["camera_a", "camera_b"] as const).map((camera, cameraIndex) => (
                <figure className="coach-video" key={camera}>
                  <figcaption>{camera === "camera_a" ? "Down the line" : "Face on"} · {finding.phase}</figcaption>
                  <video ref={registerVideo(`coach-${camera}-base`)} src={videoUrl(camera, "replay-original")} muted playsInline onLoadedMetadata={(event) => { event.currentTarget.playbackRate = speed; if (cameraIndex === 0) setDuration(event.currentTarget.duration); }} onTimeUpdate={cameraIndex === 0 ? (event) => syncFromMaster(event.currentTarget) : undefined} onError={onVideoError} />
                  <video className="coach-video__overlay" ref={registerVideo(`coach-${camera}-overlay`)} src={videoUrl(camera, "stripped")} muted playsInline onError={onVideoError} />
                </figure>
              ))}
            </div>
            <article className="coach-finding">
              <span className={`coach-finding__status status-${finding.status}`}>{finding.status}</span>
              <h3>{finding.title}</h3>
              <p>{finding.summary}</p>
              <section><h4>Why it matters</h4><p>{finding.impact}</p></section>
              <section><h4>How to fix it</h4><p>{finding.fix}</p></section>
              <section className="coach-drill"><h4>Practice drill</h4><p>{finding.drill}</p></section>
            </article>
          </div>
        )}

        {error && <div className="swing-replay__error">{error}</div>}

        <div className="swing-replay__timeline">
          <input aria-label="Replay position" type="range" min={0} max={duration || 1} step={1 / frameRate} value={Math.min(currentTime, duration || 1)} onChange={(event) => { setAllPlaying(false); seekAll(Number(event.target.value)); }} />
          <span>{currentTime.toFixed(2)}s / {duration.toFixed(2)}s</span>
        </div>

        <footer className="swing-replay__controls">
          <button onClick={() => { setAllPlaying(false); seekAll(currentTime - 1 / frameRate); }} title="Previous frame">│◀</button>
          <button className="swing-replay__play" onClick={() => setAllPlaying(!playing)}>{playing ? "Pause swing" : "Play swing"}</button>
          <button onClick={() => { setAllPlaying(false); seekAll(currentTime + 1 / frameRate); }} title="Next frame">▶│</button>
          <label>Speed<select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>{SPEEDS.map((value) => <option value={value} key={value}>{value}×</option>)}</select></label>
          {mode === "comparison" ? <button className="swing-replay__coach" onClick={() => { setMode("coach"); setTimeout(() => chooseFinding(1), 0); }}>Break it down</button> : <button className="swing-replay__flight" onClick={onPlayBallFlight}>Play ball flight →</button>}
        </footer>
      </div>
    </div>
  );
}
