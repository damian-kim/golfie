---
name: Golfie
description: Immersive telemetry HUD and 3D flight simulator
colors:
  bg: "#0b1410"
  surface: "#121c16"
  border: "#25382c"
  turf-bright: "#4cc273"
  amber: "#e8a23a"
  violet: "#8b7ce0"
  ink: "#edf2ee"
  muted: "#7c8f84"
  hud-bg: "rgba(10, 16, 13, 0.72)"
  hud-border: "rgba(76, 194, 115, 0.25)"
typography:
  display:
    fontFamily: "Manrope, system-ui, sans-serif"
    fontSize: "clamp(1.5rem, 5vw, 2.5rem)"
    fontWeight: 800
    lineHeight: 1.15
  body:
    fontFamily: "Manrope, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "JetBrains Mono, SFMono-Regular, Consolas, monospace"
    fontSize: "11px"
    fontWeight: 700
rounded:
  sm: "6px"
  md: "10px"
  lg: "16px"
components:
  metric-card:
    backgroundColor: "{colors.hud-bg}"
    rounded: "{rounded.sm}"
  layer-toggle:
    backgroundColor: "rgba(255, 255, 255, 0.03)"
    rounded: "4px"
---

# Design System: Golfie

## 1. Overview

**Creative North Star: "The Tactical Telemetry Bay"**

Golfie's visual interface is designed as an immersive, full-bleed gaming HUD overlaid on top of a 3D driving range. It balances rich telemetry with minimal, flat, and quiet aesthetics. The 3D shot viewer occupies the entire screen, while widgets float as semi-transparent panels that keep the layout clean, restrained, and dark. 

The system explicitly rejects cluttered arcade-style interfaces and overstimulating colors. Visual emphasis is achieved through color-source coding and subtle glow states rather than heavy container blocks.

**Key Characteristics:**
- **Full-bleed Viewport:** The 3D scene is the base layer and runs edge-to-edge behind overlays.
- **Glassmorphic HUD Overlays:** UI widgets float on top using semi-transparent dark backdrops and thin accent-colored borders.
- **Restrained Interaction:** Focus states and active layers use thin glowing rings rather than scale animations or heavy drop shadows.

## 2. Colors

The color palette is built on cool, dark turf-tinted neutrals with strategic neon telemetry accents that indicate the origin of the data.

### Primary
- **Turf Green** (#2e7d4f): The main anchor color representing the golf range turf.
- **Glowing Green** (#4cc273): Used for estimated or simulated trajectory lines and primary active HUD states.

### Secondary
- **Data Amber** (#e8a23a): Used for raw, real-world measured shot coordinates and sensor telemetry.
- **Model Violet** (#8b7ce0): Used for physics-fitted curves and regression model lines.

### Neutral
- **Void** (#0b1410): The deep dark-green base background of the page shell.
- **Surface** (#121c16): Base container surfaces.
- **Ink** (#edf2ee): Primary high-contrast text.
- **Muted** (#7c8f84): Secondary text and labels.

### Named Rules
**The Telemetry Source Rule.** Color coding is strictly functional: green represents simulated/estimated data, amber represents raw measured values, and violet represents mathematically fitted curves. Never mix these color roles.

## 3. Typography

**Display Font:** Manrope (sans-serif)
**Body Font:** Manrope (sans-serif)
**Label/Mono Font:** JetBrains Mono (monospace)

**Character:** Technical, clean, and legible. Monospace fonts are reserved for raw numbers, telemetry readouts, and status codes. Clean sans-serif weights handle titles and content text.

### Hierarchy
- **Display** (800, clamp(1.5rem, 5vw, 2.5rem), 1.15): Primary HUD header titles.
- **Title** (700, 14px, 1.2): Section headers and panel names.
- **Body** (400, 13px, 1.5): Labels, descriptors, and secondary text.
- **Label** (700, 11px, mono): Technical metrics and source tags.

## 4. Elevation

The elevation model relies entirely on tonal depth and layer transparency rather than heavy box shadows. Depth is flat and quiet.

### Named Rules
**The Flat-Depth Rule.** HUD panels and widgets do not use shadows. They float above the 3D canvas using dark, translucent fills (rgba(10, 16, 13, 0.72)) and thin border lines to establish depth.

## 5. Components

### Buttons
- **Shape:** Sharp, lightly rounded corners (4px).
- **Primary:** Neon green theme border, transparent background, glowing text.
- **Replay Button:** Turf green background (#2e7d4f), dark ink text. Hover scales border glow.

### Cards / Containers
- **Corner Style:** Small radius (6px).
- **Background:** Translucent glass backdrop (rgba(10, 16, 13, 0.72)).
- **Border:** Thin accent line (rgba(76, 194, 115, 0.25)).
- **Internal Padding:** 12px 14px.

### Inputs / Fields
- **Style:** Semi-transparent dark backdrops, thin borders, medium radius (10px).
- **Focus:** Thin glowing green ring.

## 6. Do's and Don'ts

### Do:
- **Do** align telemetry panels absolutely to the sides of the viewport, leaving the center of the screen open for the 3D scene.
- **Do** use JetBrains Mono for all numeric readouts to ensure clean tabular digit alignment.
- **Do** use green, amber, and violet strictly to represent the data source (estimated, measured, fitted).

### Don't:
- **Don't** use cluttered arcade-style buttons, thick progress bars, or loud icons.
- **Don't** add drop shadows or translation effects to HUD overlays.
- **Don't** use gradient text or rounded pill containers on telemetry widgets.
