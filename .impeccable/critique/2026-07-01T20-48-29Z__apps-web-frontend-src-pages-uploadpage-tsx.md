---
target: apps/web/frontend/src/pages/UploadPage.tsx
total_score: 29
p0_count: 0
p1_count: 1
timestamp: 2026-07-01T20-48-29Z
slug: apps-web-frontend-src-pages-uploadpage-tsx
---
#### Design Health Score
| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Good upload progress messages, but lacks visual percentage loader. |
| 2 | Match System / Real World | 3 | Terminology is clear and matches standard golf settings. |
| 3 | User Control and Freedom | 3 | Lacks a "Cancel" or "Go Back" link from the form layout. |
| 4 | Consistency and Standards | 3 | Standard form alignment, but uses default browser file input controls. |
| 5 | Error Prevention | 3 | Disables submission button until files are chosen, which is helpful. |
| 6 | Recognition Rather Than Recall | 3 | Labels are explicit and consistent. |
| 7 | Flexibility and Efficiency | 2 | No drag-and-drop support for file inputs. |
| 8 | Aesthetic and Minimalist Design | 3 | Clean dark theme, but slightly card-heavy configuration. |
| 9 | Error Recovery | 3 | Error banner is shown at the top of the form on failure. |
| 10 | Help and Documentation | 3 | Hints explain the recommended frame rate but lack link to docs. |
| **Total** | | **29/40** | **Good** |

#### Anti-Patterns Verdict
**LLM assessment**: The interface feels clean and on-brand, but suffers from "card fatigue" (every form section is enclosed in a standard bordered card). Using standard browser `<input type="file">` file uploads lacks premium styling and breaks the unified gaming-HUD feel.
**Deterministic scan**: No design system or color anomalies were flag-detected in the file upload layout by the automated scan.

#### Overall Impression
A solid and functional upload wizard, but it lacks visual polish on the file selection components and is missing a navigation escape route.

#### What's Working
- **Dynamic frame rate override**: Let's users specify slow-mo settings (120/240 fps) in the card UI directly.
- **Clear metadata inputs**: Straightforward form options for Club and Handedness.

#### Priority Issues
- **[P1] No Cancel/Back action**: The form does not offer a Cancel button to go back to the home page or dashboard.
  - *Why it matters*: Users are trapped on this page if they change their minds and must rely on browser history navigation.
  - *Fix*: Add a "Cancel" button alongside "Create session & upload" linking back to the home page.
  - *Suggested command*: `/impeccable layout`
- **[P2] Generic file upload UI**: Standard browser `<input type="file">` buttons look plain and unstyled.
  - *Why it matters*: It breaks the "Tactical Telemetry Bay" design language which expects custom glassmorphic styled inputs.
  - *Fix*: Replace with a custom-designed drag-and-drop zone or a stylized file button.
  - *Suggested command*: `/impeccable delight`

#### Persona Red Flags
- **Alex (Power User)**: Forced to click a file dialog twice (once for A, once for B) with no support for dragging both files simultaneously or pasting URLs.
- **Jordan (First-Timer)**: No guidance or help link if they are unsure which video is camera A (Down-the-Line) versus Camera B (Face-On).

#### Minor Observations
- The form grid could use standard margin-bottom spacing relative to the header text.
