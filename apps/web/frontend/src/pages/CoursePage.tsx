import { useState } from "react";
import "../styles/forms.css";

interface HoleData {
  number: number;
  par: number;
  yards: number;
  handicap: number;
  description: string;
}

const DEFAULT_HOLES: HoleData[] = [
  { number: 1, par: 4, yards: 385, handicap: 7, description: "Opening par 4 with a gentle dogleg right." },
  { number: 2, par: 3, yards: 165, handicap: 15, description: "Short par 3 over a scenic water hazard." },
  { number: 3, par: 5, yards: 520, handicap: 1, description: "Long double-dogleg par 5 with sand traps guarding the green." },
  { number: 4, par: 4, yards: 410, handicap: 5, description: "Tough uphill tee shot with prevailing headwinds." },
  { number: 5, par: 4, yards: 370, handicap: 11, description: "Short, drivable par 4 for aggressive players." },
  { number: 6, par: 3, yards: 195, handicap: 9, description: "Long par 3 requiring a precise mid-iron shot." },
  { number: 7, par: 5, yards: 545, handicap: 3, description: "Scenic three-shot par 5 along the lakeside rough." },
  { number: 8, par: 4, yards: 330, handicap: 17, description: "Short par 4 with a shallow bunker dividing the fairway." },
  { number: 9, par: 4, yards: 440, handicap: 13, description: "Challenging finishing hole with a narrow landing zone." },
];

export function CoursePage() {
  const [selectedHole, setSelectedHole] = useState<number>(1);
  const hole = DEFAULT_HOLES[selectedHole - 1];

  return (
    <div className="page page--focused" style={{ padding: "40px 24px" }}>
      <div>
        <h1 className="page__title">Custom 9-Hole Course</h1>
        <p className="page__subtitle">
          Map Explorer &amp; Interactive Course Layout Builder
        </p>
      </div>

      <div className="form-grid" style={{ gridTemplateColumns: "1fr 2fr", gap: "24px", marginTop: "24px" }}>
        {/* Hole Selector List */}
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <div className="card" style={{ padding: "16px" }}>
            <h2 className="card__title" style={{ fontSize: "14px", marginBottom: "16px" }}>Select Hole</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "8px" }}>
              {DEFAULT_HOLES.map((h) => (
                <button
                  key={h.number}
                  type="button"
                  className={`primary-button ${selectedHole === h.number ? "active" : ""}`}
                  style={{
                    padding: "10px 0",
                    fontSize: "13px",
                    fontWeight: "bold",
                    background: selectedHole === h.number ? "var(--color-turf-bright)" : "rgba(255,255,255,0.03)",
                    borderColor: selectedHole === h.number ? "var(--color-turf-bright)" : "var(--color-border)",
                    color: selectedHole === h.number ? "#0d0e11" : "var(--color-ink)",
                  }}
                  onClick={() => setSelectedHole(h.number)}
                >
                  Hole {h.number}
                </button>
              ))}
            </div>
          </div>

          {/* Hole Details Summary */}
          <div className="card" style={{ padding: "20px" }}>
            <h2 className="card__title" style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Hole {hole.number} Details</span>
              <span className="field__label" style={{ color: "var(--color-turf-bright)" }}>PAR {hole.par}</span>
            </h2>
            <ul style={{ listStyle: "none", padding: 0, margin: "16px 0", display: "flex", flexDirection: "column", gap: "10px" }}>
              <li style={{ display: "flex", justifyContent: "space-between", fontSize: "13.5px" }}>
                <span style={{ color: "var(--color-muted)" }}>Yardage</span>
                <span className="mono" style={{ fontWeight: "bold" }}>{hole.yards} yards</span>
              </li>
              <li style={{ display: "flex", justifyContent: "space-between", fontSize: "13.5px" }}>
                <span style={{ color: "var(--color-muted)" }}>Handicap</span>
                <span className="mono" style={{ fontWeight: "bold" }}>{hole.handicap}</span>
              </li>
            </ul>
            <p style={{ margin: "12px 0 0 0", fontSize: "13px", color: "var(--color-muted)", lineHeight: "1.5" }}>
              {hole.description}
            </p>
          </div>
        </div>

        {/* 3D Map / Scaffolding Canvas Preview Area */}
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <div
            className="card"
            style={{
              flex: 1,
              minHeight: "360px",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              background: "rgba(8, 12, 16, 0.45)",
              border: "1px dashed var(--color-border)",
              borderRadius: "var(--radius-md)",
              position: "relative",
              padding: "24px",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: "48px", marginBottom: "16px", opacity: 0.45 }}>⛳</div>
            <h3 style={{ margin: "0 0 8px 0", fontSize: "18px", color: "var(--color-ink)", fontFamily: "var(--font-display)" }}>
              Hole {hole.number} 3D Landscape Preview
            </h3>
            <p style={{ margin: 0, maxWidth: "380px", fontSize: "13px", color: "var(--color-muted)", lineHeight: "1.5" }}>
              3D interactive course contours, tees, green layouts, trees, hazards, and fairway stripes will render in this viewport once customized.
            </p>
          </div>

          {/* Design Decisions Board */}
          <div className="card" style={{ borderColor: "rgba(152, 175, 199, 0.2)" }}>
            <h2 className="card__title" style={{ fontSize: "14px", color: "var(--color-turf-bright)" }}>
              Interactive Design System Configurator
            </h2>
            <p style={{ margin: "6px 0 0 0", fontSize: "13px", color: "var(--color-muted)", lineHeight: "1.5" }}>
              Please answer the course mapping questions in the assistant conversation window to determine:
            </p>
            <ul style={{ fontSize: "12px", color: "var(--color-muted)", marginTop: "8px", paddingLeft: "20px", display: "flex", flexDirection: "column", gap: "4px" }}>
              <li>Landscape detail levels (Volumetric terrain contours, sand traps, or flat simplified vector overlays).</li>
              <li>Camera modes (Aerial map view, first-person tee view, or follow-the-ball trace mode).</li>
              <li>Custom course layout details (Fairway shape sliders, hazard adjustments, or placing bunkers dynamically).</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
