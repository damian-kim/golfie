---
target: apps/web/frontend/src/pages/CalibratePage.tsx
total_score: 31
p0_count: 0
p1_count: 1
timestamp: 2026-07-01T20-48-31Z
slug: apps-web-frontend-src-pages-calibratepage-tsx
---
#### Design Health Score
| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Explanatory text during calibration is very good. |
| 2 | Match System / Real World | 3 | Visual diagram helps understand columns and rows. |
| 3 | User Control and Freedom | 4 | Good options for custom sizes, presets, and Cancel back buttons. |
| 4 | Consistency and Standards | 2 | Uses non-design-system values for radius (4px, 8px) and background color. |
| 5 | Error Prevention | 3 | Validates uploads and parameters. |
| 6 | Recognition Rather Than Recall | 4 | The dynamic SVG diagram updates as you switch board types. |
| 7 | Flexibility and Efficiency | 3 | Good presets, though slider for marker ratio could be more responsive. |
| 8 | Aesthetic and Minimalist Design | 2 | Very dense input grid is visually busy and cluttered. |
| 9 | Error Recovery | 3 | Detailed failure messages are caught and presented. |
| 10 | Help and Documentation | 4 | Great inline hints on how to measure squares and markers. |
| **Total** | | **31/40** | **Good** |

#### Anti-Patterns Verdict
**LLM assessment**: The page has excellent functional details (the interactive SVG board diagram is a premium highlight), but the form design is overly dense and cluttered. The tabs and inputs use custom border-radii that diverge from the main project design system.
**Deterministic scan**: Flagged 5 radius anomalies in `CalibratePage.tsx` using `4px` and `8px` rounded corners (outside the standard `sm: 6px`, `md: 10px`, `lg: 16px` tokens defined in `DESIGN.md`).

#### Overall Impression
Highly functional calibration wizard with interactive visual feedback, but suffers from layout density and custom radius anomalies.

#### What's Working
- **Interactive SVG Diagram**: The `CalibrationDiagram` updates beautifully based on selected settings.
- **Clear unit conversions**: Select inputs support multiple units (mm, cm, in, m) with formatted feedback.

#### Priority Issues
- **[P1] Non-design-system border-radii**: The buttons and containers use `borderRadius: "4px"` and `"8px"` instead of the standard design system tokens.
  - *Why it matters*: Breaks visual consistency with cards and inputs in other parts of the web app.
  - *Fix*: Update styling to use standard `var(--rounded-sm)` / `6px` or `var(--rounded-md)` / `10px` tokens.
  - *Suggested command*: `/impeccable layout`
- **[P2] Visual Clutter / Grid Density**: The calibration parameters (presets, board type, dimensions) are tightly packed, causing high cognitive load.
  - *Why it matters*: The user has to digest a large number of numbers and selections simultaneously.
  - *Fix*: Organize the parameter controls into clear logical sections or use progressive disclosure.
  - *Suggested command*: `/impeccable distill`

#### Persona Red Flags
- **Sam (Accessibility-Dependent)**: The custom buttons do not use native select/focus styling, making it hard to tab through parameters.
- **Jordan (First-Timer)**: Overwhelmed by the math options (square size vs marker ratio vs marker size) without an obvious starting guide.
