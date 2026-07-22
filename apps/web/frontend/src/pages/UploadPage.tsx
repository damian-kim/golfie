import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import "../styles/forms.css";

export function UploadPage() {
  const navigate = useNavigate();
  const [club, setClub] = useState("");
  const [handedness, setHandedness] = useState<"right" | "left">("right");
  const [environment, setEnvironment] = useState<"indoor" | "outdoor" | "unknown">("indoor");
  const [ballType, setBallType] = useState("");

  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [roleA, setRoleA] = useState<"down_the_line" | "face_on">("down_the_line");
  const [roleB, setRoleB] = useState<"down_the_line" | "face_on">("face_on");
  const [fpsA, setFpsA] = useState<string>("");
  const [fpsB, setFpsB] = useState<string>("");
  const [slowMotionFactorA, setSlowMotionFactorA] = useState(1);
  const [slowMotionFactorB, setSlowMotionFactorB] = useState(1);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<string | null>(null);
  const [calibrated, setCalibrated] = useState<boolean | null>(null);

  useEffect(() => {
    api.getActiveCalibration()
      .then(() => setCalibrated(true))
      .catch(() => setCalibrated(false));
  }, []);

  const canSubmit = fileA !== null && fileB !== null && !busy;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!fileA || !fileB) return;
    setBusy(true);
    setError(null);
    try {
      setStep("Creating session…");
      const session = await api.createSession({
        environment,
        club: club || undefined,
        handedness,
        ball_type: ballType || undefined,
      });

      setStep("Uploading camera A video…");
      await api.uploadCamera(
        session.session_id,
        "camera-a",
        fileA,
        roleA,
        undefined,
        fpsA ? parseFloat(fpsA) : undefined,
        slowMotionFactorA
      );

      setStep("Uploading camera B video…");
      await api.uploadCamera(
        session.session_id,
        "camera-b",
        fileB,
        roleB,
        undefined,
        fpsB ? parseFloat(fpsB) : undefined,
        slowMotionFactorB
      );

      navigate(`/sessions/${session.session_id}/processing`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong while uploading.");
      setStep(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <div>
        <h1 className="page__title">Upload a shot</h1>
        <p className="page__subtitle">
          Two recordings of the same swing. For rendered slow-motion files, select the original
          camera capture rate—not the 30/60 fps playback rate reported by the video container.
        </p>
      </div>

      {calibrated === true && (
        <div className="status-banner status-banner--success">
          <span className="status-banner__text">✓ Cameras Calibrated (Active rig configuration found on backend)</span>
          <button type="button" className="status-banner__button" onClick={() => navigate("/calibrate")}>
            Recalibrate Setup
          </button>
        </div>
      )}

      {calibrated === false && (
        <div className="status-banner status-banner--warning">
          <span className="status-banner__text">⚠ Cameras Not Calibrated. Processing will fall back to simulated/placeholder flight data.</span>
          <button type="button" className="status-banner__button" onClick={() => navigate("/calibrate")}>
            Calibrate Cameras
          </button>
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 24 }}>
        <div className="card">
          <h2 className="card__title">Shot metadata (optional)</h2>
          <div className="form-grid">
            <label className="field">
              <span className="field__label">Club</span>
              <input
                type="text"
                placeholder="7-iron, driver, …"
                value={club}
                onChange={(e) => setClub(e.target.value)}
              />
            </label>
            <label className="field">
              <span className="field__label">Handedness</span>
              <select value={handedness} onChange={(e) => setHandedness(e.target.value as "right" | "left")}>
                <option value="right">Right-handed</option>
                <option value="left">Left-handed</option>
              </select>
            </label>
            <label className="field">
              <span className="field__label">Environment</span>
              <select
                value={environment}
                onChange={(e) => setEnvironment(e.target.value as "indoor" | "outdoor" | "unknown")}
              >
                <option value="indoor">Indoor bay</option>
                <option value="outdoor">Outdoor</option>
                <option value="unknown">Unknown</option>
              </select>
            </label>
            <label className="field">
              <span className="field__label">Ball type</span>
              <input
                type="text"
                placeholder="Pro V1, range ball, …"
                value={ballType}
                onChange={(e) => setBallType(e.target.value)}
              />
            </label>
          </div>
        </div>

        <div className="form-grid">
          <CameraUploadCard
            label="Camera A"
            file={fileA}
            onFile={setFileA}
            role={roleA}
            onRole={setRoleA}
            fps={fpsA}
            onFps={setFpsA}
            slowMotionFactor={slowMotionFactorA}
            onSlowMotionFactor={setSlowMotionFactorA}
          />
          <CameraUploadCard
            label="Camera B"
            file={fileB}
            onFile={setFileB}
            role={roleB}
            onRole={setRoleB}
            fps={fpsB}
            onFps={setFpsB}
            slowMotionFactor={slowMotionFactorB}
            onSlowMotionFactor={setSlowMotionFactorB}
          />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button type="submit" className="primary-button" disabled={!canSubmit}>
            {busy ? "Uploading…" : "Create session & upload"}
          </button>
          <button type="button" className="primary-button" style={{ borderColor: "var(--color-muted-dim)" }} onClick={() => navigate("/")} disabled={busy}>
            Cancel
          </button>
          {step && <span className="page__subtitle" style={{ color: "var(--color-turf-bright)" }}>{step}</span>}
        </div>
      </form>
    </div>
  );
}

function CameraUploadCard({
  label,
  file,
  onFile,
  role,
  onRole,
  fps,
  onFps,
  slowMotionFactor,
  onSlowMotionFactor,
}: {
  label: string;
  file: File | null;
  onFile: (f: File | null) => void;
  role: "down_the_line" | "face_on";
  onRole: (r: "down_the_line" | "face_on") => void;
  fps: string;
  onFps: (f: string) => void;
  slowMotionFactor: number;
  onSlowMotionFactor: (factor: number) => void;
}) {
  return (
    <div className="card">
      <h2 className="card__title">{label}</h2>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="field">
          <span className="field__label">Video file</span>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <label className="primary-button" style={{ 
              display: "inline-block", 
              cursor: "pointer", 
              fontSize: "12px", 
              padding: "8px 12px",
              background: file ? "rgba(232, 162, 58, 0.08)" : "transparent",
              borderColor: file ? "var(--color-amber)" : "var(--color-border)",
              color: file ? "var(--color-amber)" : "var(--color-muted)",
              borderRadius: "var(--radius-sm)"
            }}>
              {file ? "Change File" : "Choose File…"}
              <input
                type="file"
                accept="video/*"
                onChange={(e) => onFile(e.target.files?.[0] ?? null)}
                style={{ display: "none" }}
              />
            </label>
            <span style={{ fontSize: "12px", color: file ? "var(--color-ink)" : "var(--color-muted-dim)", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", flex: 1 }}>
              {file ? file.name : "No file selected"}
            </span>
          </div>
        </div>
        <label className="field">
          <span className="field__label">Placement</span>
          <select value={role} onChange={(e) => onRole(e.target.value as "down_the_line" | "face_on")}>
            <option value="down_the_line">Down-the-line</option>
            <option value="face_on">Face-on / diagonal</option>
          </select>
        </label>
        <div className="field">
          <span className="field__label">Playback slowdown</span>
          <select
            value={slowMotionFactor}
            onChange={(e) => onSlowMotionFactor(Number(e.target.value))}
          >
            <option value={1}>Normal speed (1x)</option>
            <option value={2}>2x slower</option>
            <option value={4}>4x slower</option>
            <option value={8}>8x slower</option>
            <option value={16}>16x slower</option>
          </select>
          <span className="field__hint">Choose how much longer playback is than real time.</span>
        </div>
        <div className="field">
          <span className="field__label">Capture rate override (advanced)</span>
          <div style={{ display: "flex", gap: 10, alignItems: "stretch" }}>
            <select
              value={["", "240", "120", "60", "30"].includes(fps) ? fps : "custom"}
              onChange={(e) => {
                const val = e.target.value;
                if (val === "custom") {
                  onFps("177"); // default fallback custom fps
                } else {
                  onFps(val);
                }
              }}
              style={{ flex: 1 }}
            >
              <option value="">Calculate from playback slowdown (recommended)</option>
              <option value="240">240 fps (Slow-Motion)</option>
              <option value="120">120 fps (Slow-Motion)</option>
              <option value="60">60 fps</option>
              <option value="30">30 fps (Normal)</option>
              <option value="custom">Custom...</option>
            </select>
            {!["", "240", "120", "60", "30"].includes(fps) && (
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <input
                  type="number"
                  step="any"
                  placeholder="fps"
                  value={fps}
                  onChange={(e) => onFps(e.target.value)}
                  style={{ width: "80px", textAlign: "right" }}
                />
                <span style={{ fontSize: "11px", color: "var(--color-muted)" }}>fps</span>
              </div>
            )}
          </div>
        </div>
        {file && (
          <span className="field__hint">
            {file.name} · {(file.size / (1024 * 1024)).toFixed(1)} MB. Playback slowdown survives
            even when trimming removes the phone&apos;s original slow-motion metadata.
          </span>
        )}
      </div>
    </div>
  );
}
