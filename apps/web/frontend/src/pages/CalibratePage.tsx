import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import type { CalibrationResult } from "../lib/types";
import "../styles/forms.css";

export function CalibratePage() {
  const navigate = useNavigate();
  const [boardType, setBoardType] = useState<"charuco" | "chessboard">("charuco");
  const [gridCols, setGridCols] = useState(11);
  const [gridRows, setGridRows] = useState(8);
  const [squareSize, setSquareSize] = useState(0.04);
  const [markerSize, setMarkerSize] = useState(0.03);

  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CalibrationResult | null>(null);

  const canSubmit = fileA !== null && fileB !== null && !busy;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!fileA || !fileB) return;
    setBusy(true);
    setError(null);
    setResult(null);

    try {
      const res = await api.uploadAndCalibrate(
        fileA,
        fileB,
        boardType,
        gridCols,
        gridRows,
        squareSize,
        markerSize
      );
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong during calibration.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page" style={{ maxWidth: 800, margin: "0 auto", padding: "40px 20px" }}>
      <div style={{ marginBottom: 30 }}>
        <h1 className="page__title" style={{ background: "linear-gradient(135deg, #60a5fa, #3b82f6)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Calibrate Camera Rig
        </h1>
        <p className="page__subtitle">
          Calibrate camera lenses and relative camera positions using a calibration target video.
        </p>
      </div>

      {error && (
        <div className="error-banner" style={{ borderLeft: "4px solid #ef4444", background: "rgba(239, 68, 68, 0.1)", color: "#f87171", padding: 16, borderRadius: 8, marginBottom: 20 }}>
          <h4 style={{ margin: "0 0 6px 0", fontWeight: "bold" }}>Calibration Failed</h4>
          <p style={{ margin: 0, fontSize: "0.95em" }}>{error}</p>
        </div>
      )}

      {result && (
        <div className="card" style={{ border: "1px solid #10b981", background: "rgba(16, 185, 129, 0.05)", borderRadius: 12, padding: 24, marginBottom: 30 }}>
          <h2 className="card__title" style={{ color: "#10b981", display: "flex", alignItems: "center", gap: 8 }}>
            ✓ Rig Calibrated Successfully
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 16 }}>
            <div>
              <span className="field__label" style={{ color: "#9ca3af" }}>Reprojection Error</span>
              <div style={{ fontSize: "1.8em", fontWeight: "bold", color: "#f3f4f6" }}>
                {result.reprojection_error_px?.toFixed(3)} <span style={{ fontSize: "0.5em", fontWeight: "normal" }}>px</span>
              </div>
            </div>
            <div>
              <span className="field__label" style={{ color: "#9ca3af" }}>Confidence Score</span>
              <div style={{ fontSize: "1.8em", fontWeight: "bold", color: "#f3f4f6" }}>
                {Math.round(result.confidence * 100)}%
              </div>
            </div>
          </div>

          <div style={{ marginTop: 24, padding: "12px 0 0 0", borderTop: "1px solid rgba(255,255,255,0.1)", display: "flex", gap: 14 }}>
            <button className="primary-button" onClick={() => navigate("/")}>
              Continue to upload shot →
            </button>
            <button className="primary-button" style={{ background: "transparent", border: "1px solid rgba(255,255,255,0.2)" }} onClick={() => setResult(null)}>
              Recalibrate
            </button>
          </div>
        </div>
      )}

      {!result && (
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {/* Setup Walkthrough Card */}
          <div className="card" style={{ border: "1px solid rgba(59, 130, 246, 0.2)", background: "rgba(59, 130, 246, 0.03)" }}>
            <h2 className="card__title" style={{ color: "#60a5fa" }}>Setup Instructions</h2>
            <ol style={{ paddingLeft: 20, margin: "10px 0 0 0", display: "flex", flexDirection: "column", gap: 10, color: "#d1d5db" }}>
              <li>
                Print a <strong>ChArUco</strong> (recommended) or <strong>Chessboard</strong> calibration board and mount it flat.
              </li>
              <li>
                Place your tripods in their final positions (Camera A Down-the-line, Camera B Face-on). <strong>Do not move them</strong> until you finish hitting your golf shots!
              </li>
              <li>
                Start recording on both cameras. Wave the board slowly in the common overlap view (where the ball is hit), holding it at various angles and rotations so both cameras capture it.
              </li>
              <li>
                Upload the calibration recordings below to automatically sync, extract frames, and compute the camera rig geometry.
              </li>
            </ol>
          </div>

          {/* Pattern settings */}
          <div className="card">
            <h2 className="card__title">Calibration target parameters</h2>
            <div className="form-grid">
              <label className="field">
                <span className="field__label">Target Type</span>
                <select value={boardType} onChange={(e) => setBoardType(e.target.value as any)}>
                  <option value="charuco">ChArUco board (checkerboard + markers)</option>
                  <option value="chessboard">Standard Chessboard</option>
                </select>
              </label>
              <label className="field">
                <span className="field__label">Grid Dimensions (cols x rows)</span>
                <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <input
                    type="number"
                    value={gridCols}
                    onChange={(e) => setGridCols(parseInt(e.target.value) || 0)}
                    placeholder="columns"
                  />
                  <span>x</span>
                  <input
                    type="number"
                    value={gridRows}
                    onChange={(e) => setGridRows(parseInt(e.target.value) || 0)}
                    placeholder="rows"
                  />
                </div>
              </label>
              <label className="field">
                <span className="field__label">Square Size (meters)</span>
                <input
                  type="number"
                  step="0.001"
                  value={squareSize}
                  onChange={(e) => setSquareSize(parseFloat(e.target.value) || 0)}
                />
              </label>
              {boardType === "charuco" && (
                <label className="field">
                  <span className="field__label">Marker Size (meters)</span>
                  <input
                    type="number"
                    step="0.001"
                    value={markerSize}
                    onChange={(e) => setMarkerSize(parseFloat(e.target.value) || 0)}
                  />
                </label>
              )}
            </div>
          </div>

          {/* Video uploads */}
          <div className="form-grid">
            <div className="card">
              <h2 className="card__title">Camera A Video (Down-the-Line)</h2>
              <label className="field">
                <span className="field__label">Select calibration clip</span>
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setFileA(e.target.files?.[0] ?? null)}
                />
              </label>
              {fileA && (
                <span className="field__hint">
                  {fileA.name} · {(fileA.size / (1024 * 1024)).toFixed(1)} MB
                </span>
              )}
            </div>
            <div className="card">
              <h2 className="card__title">Camera B Video (Face-On)</h2>
              <label className="field">
                <span className="field__label">Select calibration clip</span>
                <input
                  type="file"
                  accept="video/*"
                  onChange={(e) => setFileB(e.target.files?.[0] ?? null)}
                />
              </label>
              {fileB && (
                <span className="field__hint">
                  {fileB.name} · {(fileB.size / (1024 * 1024)).toFixed(1)} MB
                </span>
              )}
            </div>
          </div>

          {/* Submission and loading state */}
          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            <button type="submit" className="primary-button" disabled={!canSubmit}>
              {busy ? "Processing Calibration…" : "Run Calibration"}
            </button>
            <button type="button" className="primary-button" style={{ background: "transparent", border: "1px solid rgba(255,255,255,0.2)" }} onClick={() => navigate("/")}>
              Cancel
            </button>
            {busy && (
              <span className="page__subtitle" style={{ margin: 0, animation: "pulse 1.5s infinite" }}>
                Analyzing video files, extracting synced frames, detecting targets, and calculating camera matrix offsets…
              </span>
            )}
          </div>
        </form>
      )}
    </div>
  );
}
