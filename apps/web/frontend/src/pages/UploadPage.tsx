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
      await api.uploadCamera(session.session_id, "camera-a", fileA, roleA);

      setStep("Uploading camera B video…");
      await api.uploadCamera(session.session_id, "camera-b", fileB, roleB);

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
          Two synced iPhone recordings of the same swing. 1080p at 240fps is recommended for
          reliable ball tracking once detection is implemented.
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
          />
          <CameraUploadCard
            label="Camera B"
            file={fileB}
            onFile={setFileB}
            role={roleB}
            onRole={setRoleB}
          />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button type="submit" className="primary-button" disabled={!canSubmit}>
            {busy ? "Uploading…" : "Create session & upload"}
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
}: {
  label: string;
  file: File | null;
  onFile: (f: File | null) => void;
  role: "down_the_line" | "face_on";
  onRole: (r: "down_the_line" | "face_on") => void;
}) {
  return (
    <div className="card">
      <h2 className="card__title">{label}</h2>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <label className="field">
          <span className="field__label">Video file</span>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => onFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <label className="field">
          <span className="field__label">Placement</span>
          <select value={role} onChange={(e) => onRole(e.target.value as "down_the_line" | "face_on")}>
            <option value="down_the_line">Down-the-line</option>
            <option value="face_on">Face-on / diagonal</option>
          </select>
        </label>
        {file && (
          <span className="field__hint">
            {file.name} · {(file.size / (1024 * 1024)).toFixed(1)} MB
          </span>
        )}
      </div>
    </div>
  );
}
