import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../lib/api";
import type { CalibrationResult } from "../lib/types";
import "../styles/forms.css";

export interface BoardPreset {
  id: string;
  name: string;
  boardType: "charuco" | "chessboard";
  gridCols: number;
  gridRows: number;
  squareSize: number;
  markerSize?: number;
  unit: "mm" | "cm" | "in" | "m";
}

const PRESETS: BoardPreset[] = [
  {
    id: "charuco_a4_std",
    name: "Standard A4 ChArUco (11x8 grid, 20mm squares)",
    boardType: "charuco",
    gridCols: 11,
    gridRows: 8,
    squareSize: 20,
    markerSize: 15,
    unit: "mm",
  },
  {
    id: "charuco_letter_std",
    name: "Standard Letter ChArUco (11x8 grid, 25mm squares)",
    boardType: "charuco",
    gridCols: 11,
    gridRows: 8,
    squareSize: 25,
    markerSize: 19,
    unit: "mm",
  },
  {
    id: "chessboard_a4_std",
    name: "Standard A4 Chessboard (9x6 grid, 25mm squares)",
    boardType: "chessboard",
    gridCols: 9,
    gridRows: 6,
    squareSize: 25,
    unit: "mm",
  },
  {
    id: "chessboard_poster_std",
    name: "Poster Chessboard (9x7 grid, 50mm squares)",
    boardType: "chessboard",
    gridCols: 9,
    gridRows: 7,
    squareSize: 50,
    unit: "mm",
  },
  {
    id: "custom",
    name: "Custom Parameters...",
    boardType: "charuco",
    gridCols: 11,
    gridRows: 8,
    squareSize: 40,
    markerSize: 30,
    unit: "mm",
  },
];

const UNIT_MULTIPLIERS: Record<string, number> = {
  mm: 0.001,
  cm: 0.01,
  in: 0.0254,
  m: 1.0,
};

function CalibrationDiagram({ boardType }: { boardType: "charuco" | "chessboard" }) {
  const isCharuco = boardType === "charuco";
  
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10, alignItems: "center" }}>
      <svg viewBox="0 0 240 180" style={{ width: "100%", height: "auto", background: "rgba(10, 16, 13, 0.4)", borderRadius: "8px", border: "1px solid rgba(76, 194, 115, 0.15)", padding: "10px" }}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-turf-bright)" />
          </marker>
          <marker id="arrow-subtle" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--color-muted)" />
          </marker>
        </defs>

        {/* Draw board grid (4x3 squares for simple display) */}
        <g transform="translate(45, 45)">
          {/* Row 0 */}
          <rect x="0" y="0" width="30" height="30" fill="#2d3530" />
          <rect x="30" y="0" width="30" height="30" fill="#d2ded6" />
          <rect x="60" y="0" width="30" height="30" fill="#2d3530" />
          <rect x="90" y="0" width="30" height="30" fill="#d2ded6" />

          {/* Row 1 */}
          <rect x="0" y="30" width="30" height="30" fill="#d2ded6" />
          <rect x="30" y="30" width="30" height="30" fill="#2d3530" />
          <rect x="60" y="30" width="30" height="30" fill="#d2ded6" />
          <rect x="90" y="30" width="30" height="30" fill="#2d3530" />

          {/* Row 2 */}
          <rect x="0" y="60" width="30" height="30" fill="#2d3530" />
          <rect x="30" y="60" width="30" height="30" fill="#d2ded6" />
          <rect x="60" y="60" width="30" height="30" fill="#2d3530" />
          <rect x="90" y="60" width="30" height="30" fill="#d2ded6" />

          {/* ChArUco markers in white squares */}
          {isCharuco && (
            <>
              {/* Markers on Row 0 (col 1, 3) */}
              <rect x="37.5" y="7.5" width="15" height="15" fill="#111" rx="1" />
              <rect x="42" y="12" width="6" height="6" fill="#d2ded6" />
              <rect x="97.5" y="7.5" width="15" height="15" fill="#111" rx="1" />
              <rect x="102" y="12" width="6" height="6" fill="#d2ded6" />
              
              {/* Markers on Row 1 (col 0, 2) */}
              <rect x="7.5" y="37.5" width="15" height="15" fill="#111" rx="1" />
              <rect x="12" y="42" width="6" height="6" fill="#d2ded6" />
              <rect x="67.5" y="37.5" width="15" height="15" fill="#111" rx="1" />
              <rect x="72" y="42" width="6" height="6" fill="#d2ded6" />

              {/* Markers on Row 2 (col 1, 3) */}
              <rect x="37.5" y="67.5" width="15" height="15" fill="#111" rx="1" />
              <rect x="42" y="72" width="6" height="6" fill="#d2ded6" />
              <rect x="97.5" y="67.5" width="15" height="15" fill="#111" rx="1" />
              <rect x="102" y="72" width="6" height="6" fill="#d2ded6" />
            </>
          )}
        </g>

        {/* Grid count labels */}
        {/* Columns (width) label at the top */}
        <line x1="45" y1="22" x2="165" y2="22" stroke="var(--color-turf-bright)" strokeWidth="1.5" markerStart="url(#arrow)" markerEnd="url(#arrow)" />
        <text x="105" y="14" fill="var(--color-turf-bright)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle" fontWeight="bold">Cols (4 squares across)</text>

        {/* Rows (height) label on the left */}
        <line x1="22" y1="45" x2="22" y2="135" stroke="var(--color-turf-bright)" strokeWidth="1.5" markerStart="url(#arrow)" markerEnd="url(#arrow)" />
        <text x="14" y="90" fill="var(--color-turf-bright)" fontSize="9" fontFamily="var(--font-mono)" textAnchor="middle" fontWeight="bold" transform="rotate(-90 14 90)">Rows (3 squares down)</text>

        {/* Dimension labels for sizes */}
        {/* Square size */}
        <line x1="168" y1="135" x2="168" y2="105" stroke="var(--color-muted)" strokeWidth="1.2" markerStart="url(#arrow-subtle)" markerEnd="url(#arrow-subtle)" />
        <text x="174" y="124" fill="var(--color-muted)" fontSize="9" fontFamily="var(--font-mono)" dominantBaseline="middle">Square Size (S)</text>

        {/* Marker size */}
        {isCharuco && (
          <g>
            <line x1="142.5" y1="75" x2="142.5" y2="90" stroke="rgba(255,255,255,0.7)" strokeWidth="1" markerStart="url(#arrow-subtle)" markerEnd="url(#arrow-subtle)" />
            <path d="M 142.5 82.5 L 175 75" stroke="var(--color-muted)" strokeWidth="0.8" strokeDasharray="2,2" />
            <text x="178" y="77" fill="var(--color-muted)" fontSize="9" fontFamily="var(--font-mono)" dominantBaseline="middle">Marker Size (M)</text>
          </g>
        )}
      </svg>
      <div style={{ fontSize: "11px", color: "var(--color-muted-dim)", textAlign: "center", lineHeight: "1.4" }}>
        Measure the physical target with a ruler. Columns & Rows are the count of alternating black/white squares.
      </div>
    </div>
  );
}

export function CalibratePage() {
  const navigate = useNavigate();

  // Preset state
  const [selectedPresetId, setSelectedPresetId] = useState<string>("charuco_a4_std");

  // target parameters state
  const [boardType, setBoardType] = useState<"charuco" | "chessboard">("charuco");
  const [gridCols, setGridCols] = useState(11);
  const [gridRows, setGridRows] = useState(8);
  const [squareSize, setSquareSize] = useState(20);
  const [unit, setUnit] = useState<"mm" | "cm" | "in" | "m">("mm");

  // Marker states
  const [markerMode, setMarkerMode] = useState<"ratio" | "custom">("ratio");
  const [markerRatio, setMarkerRatio] = useState(75); // percent
  const [markerSize, setMarkerSize] = useState(15);

  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CalibrationResult | null>(null);

  // Auto set preset parameters
  const handlePresetChange = (presetId: string) => {
    setSelectedPresetId(presetId);
    if (presetId === "custom") return;

    const preset = PRESETS.find((p) => p.id === presetId);
    if (preset) {
      setBoardType(preset.boardType);
      setGridCols(preset.gridCols);
      setGridRows(preset.gridRows);
      setSquareSize(preset.squareSize);
      setUnit(preset.unit);
      if (preset.markerSize !== undefined) {
        setMarkerMode("custom");
        setMarkerSize(preset.markerSize);
      } else {
        setMarkerMode("ratio");
        setMarkerRatio(75);
      }
    }
  };

  const markAsCustom = () => {
    setSelectedPresetId("custom");
  };

  const getSubmitValues = () => {
    const mult = UNIT_MULTIPLIERS[unit];
    const sqMeters = squareSize * mult;
    let markMeters = 0.03;
    if (boardType === "charuco") {
      if (markerMode === "ratio") {
        markMeters = sqMeters * (markerRatio / 100);
      } else {
        markMeters = markerSize * mult;
      }
    }
    return {
      squareSizeMeters: sqMeters,
      markerSizeMeters: markMeters,
    };
  };

  const formatMeters = (val: number) => {
    const mult = UNIT_MULTIPLIERS[unit];
    const metersVal = val * mult;
    if (isNaN(metersVal)) return "0 m";
    return `${metersVal.toFixed(4).replace(/\.?0+$/, "")} m`;
  };

  const calculatedMarkerSize = markerMode === "ratio" ? squareSize * (markerRatio / 100) : markerSize;

  const canSubmit = fileA !== null && fileB !== null && !busy;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!fileA || !fileB) return;
    setBusy(true);
    setError(null);
    setResult(null);

    const { squareSizeMeters, markerSizeMeters } = getSubmitValues();

    try {
      const res = await api.uploadAndCalibrate(
        fileA,
        fileB,
        boardType,
        gridCols,
        gridRows,
        squareSizeMeters,
        markerSizeMeters
      );
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong during calibration.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page page--focused" style={{ padding: "40px 24px" }}>
      <div style={{ marginBottom: 30 }}>
        <h1 className="page__title">
          Calibrate Camera Rig
        </h1>
        <p className="page__subtitle">
          Calibrate camera lenses and relative camera positions using a calibration target video.
        </p>
      </div>

      {error && (
        <div className="error-banner" style={{ marginBottom: 20 }}>
          <h4 style={{ margin: "0 0 6px 0", fontWeight: "bold" }}>Calibration Failed</h4>
          <p style={{ margin: 0, fontSize: "0.95em" }}>{error}</p>
        </div>
      )}

      {result && (
        <div className="card" style={{ borderColor: "rgba(76, 194, 115, 0.45)", background: "rgba(10, 16, 13, 0.85)", padding: 24, marginBottom: 30 }}>
          <h2 className="card__title" style={{ color: "var(--color-turf-bright)", display: "flex", alignItems: "center", gap: 8 }}>
            ✓ Rig Calibrated Successfully
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 16 }}>
            <div>
              <span className="field__label" style={{ color: "var(--color-muted)" }}>Reprojection Error</span>
              <div style={{ fontSize: "1.8em", fontWeight: "bold", fontFamily: "var(--font-mono)", color: "var(--color-ink)" }}>
                {result.reprojection_error_px?.toFixed(3)} <span style={{ fontSize: "0.5em", fontWeight: "normal", color: "var(--color-muted)" }}>px</span>
              </div>
            </div>
            <div>
              <span className="field__label" style={{ color: "var(--color-muted)" }}>Confidence Score</span>
              <div style={{ fontSize: "1.8em", fontWeight: "bold", fontFamily: "var(--font-mono)", color: "var(--color-ink)" }}>
                {Math.round(result.confidence * 100)}%
              </div>
            </div>
          </div>

          <div style={{ marginTop: 24, padding: "16px 0 0 0", borderTop: "1px solid rgba(255,255,255,0.06)", display: "flex", gap: 14 }}>
            <button className="primary-button" onClick={() => navigate("/")}>
              Continue to upload shot →
            </button>
            <button className="primary-button" style={{ borderColor: "var(--color-muted)" }} onClick={() => setResult(null)}>
              Recalibrate
            </button>
          </div>
        </div>
      )}

      {!result && (
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          {/* Setup Walkthrough Card */}
          <div className="card" style={{ borderColor: "rgba(76, 194, 115, 0.2)", background: "rgba(10, 16, 13, 0.72)" }}>
            <h2 className="card__title" style={{ color: "var(--color-turf-bright)" }}>Setup Instructions</h2>
            <ol style={{ paddingLeft: 20, margin: "10px 0 0 0", display: "flex", flexDirection: "column", gap: 10, color: "var(--color-muted)", fontSize: "13px", lineHeight: "1.5" }}>
              <li>
                Print a <strong>ChArUco</strong> (recommended) or <strong>Chessboard</strong> calibration board and mount it flat. 
                {" "}<a href="https://calib.io/pages/camera-calibration-pattern-generator" target="_blank" rel="noreferrer" style={{ color: "var(--color-turf-bright)", textDecoration: "underline" }}>Download / Generate printable calibration board</a>
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
            
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Preset Selector */}
              <label className="field">
                <span className="field__label">Calibration Board Preset</span>
                <select value={selectedPresetId} onChange={(e) => handlePresetChange(e.target.value)}>
                  {PRESETS.map((preset) => (
                    <option key={preset.id} value={preset.id}>
                      {preset.name}
                    </option>
                  ))}
                </select>
              </label>

              {/* Grid Layout: Left is inputs, Right is visual aid */}
              <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 30, alignItems: "start" }}>
                {/* Inputs Column */}
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                  
                  {/* Board Type Toggle */}
                  <div className="field">
                    <span className="field__label">Board Type</span>
                    <div style={{ display: "flex", gap: 10 }}>
                      <button
                        type="button"
                        onClick={() => { setBoardType("charuco"); markAsCustom(); }}
                        style={{
                          flex: 1,
                          padding: "10px",
                          background: boardType === "charuco" ? "rgba(76, 194, 115, 0.12)" : "rgba(10, 16, 13, 0.85)",
                          border: boardType === "charuco" ? "1px solid var(--color-turf-bright)" : "1px solid var(--color-border)",
                          color: boardType === "charuco" ? "var(--color-turf-bright)" : "var(--color-muted)",
                          borderRadius: "4px",
                          fontWeight: "bold",
                          cursor: "pointer",
                          transition: "all 0.15s ease",
                          fontFamily: "var(--font-display)",
                          fontSize: "12px",
                          textTransform: "uppercase",
                          letterSpacing: "0.05em",
                        }}
                      >
                        ChArUco Board
                      </button>
                      <button
                        type="button"
                        onClick={() => { setBoardType("chessboard"); markAsCustom(); }}
                        style={{
                          flex: 1,
                          padding: "10px",
                          background: boardType === "chessboard" ? "rgba(76, 194, 115, 0.12)" : "rgba(10, 16, 13, 0.85)",
                          border: boardType === "chessboard" ? "1px solid var(--color-turf-bright)" : "1px solid var(--color-border)",
                          color: boardType === "chessboard" ? "var(--color-turf-bright)" : "var(--color-muted)",
                          borderRadius: "4px",
                          fontWeight: "bold",
                          cursor: "pointer",
                          transition: "all 0.15s ease",
                          fontFamily: "var(--font-display)",
                          fontSize: "12px",
                          textTransform: "uppercase",
                          letterSpacing: "0.05em",
                        }}
                      >
                        Chessboard
                      </button>
                    </div>
                  </div>

                  {/* Grid Dimensions */}
                  <label className="field">
                    <span className="field__label">Grid Dimensions (Columns x Rows)</span>
                    <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                      <input
                        type="number"
                        value={gridCols}
                        onChange={(e) => { setGridCols(parseInt(e.target.value) || 0); markAsCustom(); }}
                        placeholder="columns"
                        style={{ flex: 1 }}
                      />
                      <span style={{ color: "var(--color-muted)" }}>x</span>
                      <input
                        type="number"
                        value={gridRows}
                        onChange={(e) => { setGridRows(parseInt(e.target.value) || 0); markAsCustom(); }}
                        placeholder="rows"
                        style={{ flex: 1 }}
                      />
                    </div>
                  </label>

                  {/* Square Size */}
                  <label className="field">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <span className="field__label">Square Size</span>
                      <span className="field__hint" style={{ color: "var(--color-turf-bright)" }}>
                        (= {formatMeters(squareSize)})
                      </span>
                    </div>
                    <div style={{ display: "flex", gap: 6, alignItems: "stretch", width: "100%" }}>
                      <input
                        type="number"
                        step="any"
                        value={squareSize}
                        onChange={(e) => { setSquareSize(parseFloat(e.target.value) || 0); markAsCustom(); }}
                        style={{ flex: 1 }}
                      />
                      <select
                        value={unit}
                        onChange={(e) => { setUnit(e.target.value as any); markAsCustom(); }}
                        style={{ width: "80px", minWidth: "80px", flexShrink: 0, paddingRight: "24px" }}
                      >
                        <option value="mm">mm</option>
                        <option value="cm">cm</option>
                        <option value="in">inches</option>
                        <option value="m">m</option>
                      </select>
                    </div>
                  </label>

                  {/* Marker Size (ChArUco Only) */}
                  {boardType === "charuco" && (
                    <div className="field">
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span className="field__label">Marker Size</span>
                        <span className="field__hint" style={{ color: "var(--color-turf-bright)" }}>
                          (= {formatMeters(calculatedMarkerSize)})
                        </span>
                      </div>
                      
                      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                        <div style={{ display: "flex", gap: 10 }}>
                          <button
                            type="button"
                            onClick={() => { setMarkerMode("ratio"); markAsCustom(); }}
                            style={{
                              flex: 1,
                              padding: "8px",
                              background: markerMode === "ratio" ? "rgba(76, 194, 115, 0.08)" : "rgba(10, 16, 13, 0.85)",
                              border: markerMode === "ratio" ? "1px solid var(--color-turf-bright)" : "1px solid var(--color-border)",
                              color: markerMode === "ratio" ? "var(--color-turf-bright)" : "var(--color-muted)",
                              borderRadius: "4px",
                              fontWeight: "600",
                              cursor: "pointer",
                              transition: "all 0.15s ease",
                              fontFamily: "var(--font-display)",
                              fontSize: "11px",
                            }}
                          >
                            Standard Ratio (75%)
                          </button>
                          <button
                            type="button"
                            onClick={() => { setMarkerMode("custom"); markAsCustom(); }}
                            style={{
                              flex: 1,
                              padding: "8px",
                              background: markerMode === "custom" ? "rgba(76, 194, 115, 0.08)" : "rgba(10, 16, 13, 0.85)",
                              border: markerMode === "custom" ? "1px solid var(--color-turf-bright)" : "1px solid var(--color-border)",
                              color: markerMode === "custom" ? "var(--color-turf-bright)" : "var(--color-muted)",
                              borderRadius: "4px",
                              fontWeight: "600",
                              cursor: "pointer",
                              transition: "all 0.15s ease",
                              fontFamily: "var(--font-display)",
                              fontSize: "11px",
                            }}
                          >
                            Custom Size
                          </button>
                        </div>
                        {markerMode === "ratio" ? (
                          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                            <input
                              type="range"
                              min="50"
                              max="90"
                              value={markerRatio}
                              onChange={(e) => { setMarkerRatio(parseInt(e.target.value)); markAsCustom(); }}
                              style={{ flex: 1, accentColor: "var(--color-turf-bright)" }}
                            />
                            <span style={{ fontSize: "12px", fontFamily: "var(--font-mono)", color: "var(--color-ink)", width: "35px", textAlign: "right" }}>
                              {markerRatio}%
                            </span>
                          </div>
                        ) : (
                          <div style={{ display: "flex", gap: 6, alignItems: "stretch" }}>
                            <input
                              type="number"
                              step="any"
                              value={markerSize}
                              onChange={(e) => { setMarkerSize(parseFloat(e.target.value) || 0); markAsCustom(); }}
                              style={{ flex: 1 }}
                            />
                            <select
                              value={unit}
                              onChange={(e) => { setUnit(e.target.value as any); markAsCustom(); }}
                              style={{ width: "80px", minWidth: "80px", flexShrink: 0, paddingRight: "24px" }}
                            >
                              <option value="mm">mm</option>
                              <option value="cm">cm</option>
                              <option value="in">inches</option>
                              <option value="m">m</option>
                            </select>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                </div>

                {/* Visual Aid Column */}
                <div>
                  <CalibrationDiagram boardType={boardType} />
                </div>
              </div>
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
            <button type="button" className="primary-button" style={{ borderColor: "var(--color-muted-dim)" }} onClick={() => navigate("/")}>
              Cancel
            </button>
            {busy && (
              <span className="page__subtitle" style={{ margin: 0, color: "var(--color-turf-bright)" }}>
                Analyzing video files, extracting synced frames, detecting targets, and calculating camera matrix offsets…
              </span>
            )}
          </div>
        </form>
      )}
    </div>
  );
}
