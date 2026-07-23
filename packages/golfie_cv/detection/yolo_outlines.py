import cv2
import numpy as np
import subprocess
import tempfile
import json
from pathlib import Path
from golfie_cv.sync import _find_ffmpeg_fallback

def render_stripped_outlines_video(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    output_path: Path,
    background_model: np.ndarray | None = None,
    ball_track_2d: list | None = None,
    model=None,
    pose_model=None,
    inference_stride: int = 2,
    inference_size: int = 384,
    comparison_output_path: Path | None = None,
    person_only: bool = True,
    pose_analysis_output_path: Path | None = None,
    impact_frame: int | None = None,
    camera_role: str = "unknown",
    handedness: str = "right",
) -> None:
    """Run YOLOv8 segmentation on the relevant frame range of the video.
    
    Draws glowing neon-colored outlines of the person (cyan) and tracked ball (green)
    on a black background. The default person-only path deliberately skips the expensive,
    unreliable Canny/Hough club solver; it remains available for diagnostic renders.
    Transcodes the final output to H.264 MP4 and logs the frame-timestamp mapping JSON.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    # 1. Load YOLOv8 segmentation model
    if model is None:
        from ultralytics import YOLO
        model = YOLO("yolov8n-seg.pt")
    inference_stride = max(1, int(inference_stride))
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Map 2D ball track list to a lookup dict {frame_index: (x, y, radius)}
    ball_lookup = {}
    if ball_track_2d:
        for pt in ball_track_2d:
            f_idx = getattr(pt, "frame_index", None)
            if f_idx is None and isinstance(pt, dict):
                f_idx = pt.get("frame_index")
                x = pt.get("x_px")
                y = pt.get("y_px")
                r = pt.get("radius_px", 4)
            else:
                x = getattr(pt, "x_px", None)
                y = getattr(pt, "y_px", None)
                r = getattr(pt, "radius_px", 4)
            if f_idx is not None and x is not None and y is not None:
                ball_lookup[int(f_idx)] = (float(x), float(y), float(r))

    # Helper function to draw glowing outline
    def draw_glowing_poly(img, poly, color, thickness=2):
        poly_int = np.array(poly, dtype=np.int32)
        glow_color = [max(0, int(c * 0.35)) for c in color]
        cv2.polylines(img, [poly_int], isClosed=True, color=glow_color, thickness=thickness + 4, lineType=cv2.LINE_AA)
        cv2.polylines(img, [poly_int], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Helper function to draw glowing circle
    def draw_glowing_circle(img, center, radius, color):
        glow_color = [max(0, int(c * 0.35)) for c in color]
        cv2.circle(img, center, int(radius) + 4, glow_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, center, max(2, int(radius)), color, thickness=-1, lineType=cv2.LINE_AA)

    def draw_glowing_line(img, start, end, color, thickness=3):
        glow_color = [max(0, int(c * 0.3)) for c in color]
        cv2.line(img, start, end, glow_color, thickness=thickness + 5, lineType=cv2.LINE_AA)
        cv2.line(img, start, end, color, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_pose_overlay(img, pose_results) -> None:
        """Draw COCO pose joints as swing-analysis guides."""
        if not pose_results or pose_results[0].keypoints is None:
            return
        keypoints = pose_results[0].keypoints
        if keypoints.xy is None or len(keypoints.xy) == 0:
            return
        xy = keypoints.xy[0].detach().cpu().numpy()
        confidence = (
            keypoints.conf[0].detach().cpu().numpy()
            if keypoints.conf is not None
            else np.ones(len(xy), dtype=float)
        )

        def point(index: int):
            if index >= len(xy) or confidence[index] < 0.28:
                return None
            x, y = xy[index]
            if not np.isfinite(x) or not np.isfinite(y):
                return None
            return int(round(x)), int(round(y))

        def segment(a: int, b: int, color, thickness=3):
            start, end = point(a), point(b)
            if start is not None and end is not None:
                draw_glowing_line(img, start, end, color, thickness)

        # Left/right refer to the detected person's anatomy. A mirrored phone
        # export will therefore make them appear reversed on screen.
        left_arm = (255, 75, 255)       # magenta
        right_arm = (0, 165, 255)       # amber
        shoulder_axis = (255, 255, 0)   # cyan
        hip_axis = (80, 255, 120)       # green
        legs = (210, 150, 75)           # blue
        balance = (245, 245, 245)       # white

        segment(5, 7, left_arm, 4)
        segment(7, 9, left_arm, 4)
        segment(6, 8, right_arm, 4)
        segment(8, 10, right_arm, 4)
        segment(5, 6, shoulder_axis, 3)
        segment(5, 11, shoulder_axis, 2)
        segment(6, 12, shoulder_axis, 2)
        segment(11, 12, hip_axis, 3)
        segment(11, 13, legs, 3)
        segment(13, 15, legs, 3)
        segment(12, 14, legs, 3)
        segment(14, 16, legs, 3)

        shoulders = (point(5), point(6))
        hips = (point(11), point(12))
        ankles = (point(15), point(16))
        if all(p is not None for p in shoulders + hips):
            mid_shoulder = tuple(np.rint(np.mean(np.asarray(shoulders), axis=0)).astype(int))
            mid_hip = tuple(np.rint(np.mean(np.asarray(hips), axis=0)).astype(int))
            draw_glowing_line(img, mid_shoulder, mid_hip, balance, 2)
            draw_glowing_circle(img, mid_hip, 4, balance)
            if all(p is not None for p in ankles):
                stance_y = int(round((ankles[0][1] + ankles[1][1]) * 0.5))
                cv2.line(img, mid_hip, (mid_hip[0], stance_y), balance, 1, cv2.LINE_AA)

        for index in range(5, 17):
            joint = point(index)
            if joint is not None:
                draw_glowing_circle(img, joint, 3, balance)

    frame_mappings = []
    pose_observations = []

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_avi = Path(tmpdir) / "temp_outlines.avi"
        temp_original_avi = Path(tmpdir) / "temp_original.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_writer = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height))
        original_writer = (
            cv2.VideoWriter(str(temp_original_avi), fourcc, fps, (width, height))
            if comparison_output_path is not None else None
        )
        
        # Sequential decode is reliable for phone H.264/B-frame streams.
        for _ in range(start_frame):
            if not cap.grab():
                break

        previous_canvas = None
        previous_gray = None
        previous_club_line = None
        
        for frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if original_writer is not None:
                original_writer.write(frame)
                
            # Log exact frame mapping and timestamp matching VFR standards
            cropped_idx = frame_idx - start_frame
            frame_mappings.append({
                "original_frame_idx": frame_idx,
                "cropped_frame_idx": cropped_idx,
                "original_timestamp_sec": round(frame_idx / fps, 5),
                "cropped_timestamp_sec": round(cropped_idx / fps, 5)
            })

            # At 240 captured fps, adjacent masks are nearly identical. Run
            # inference at a bounded stride and hold the previous outline on
            # intervening frames to halve render latency without changing the
            # output timeline.
            should_infer = previous_canvas is None or cropped_idx % inference_stride == 0
            if not should_infer:
                out_writer.write(previous_canvas)
                continue

            # Run YOLOv8 segmentation on the current frame
            results = model(
                frame,
                verbose=False,
                imgsz=max(320, int(inference_size)),
                classes=[0, 32],
                max_det=3,
            )
            pose_results = None
            if pose_model is not None:
                try:
                    pose_results = pose_model(
                        frame,
                        verbose=False,
                        imgsz=max(320, int(inference_size)),
                        classes=[0],
                        max_det=1,
                    )
                    if (
                        pose_results
                        and pose_results[0].keypoints is not None
                        and pose_results[0].keypoints.xy is not None
                        and len(pose_results[0].keypoints.xy) > 0
                    ):
                        keypoints = pose_results[0].keypoints
                        xy = keypoints.xy[0].detach().cpu().numpy()
                        confidence = (
                            keypoints.conf[0].detach().cpu().numpy()
                            if keypoints.conf is not None
                            else np.ones(len(xy), dtype=float)
                        )
                        pose_observations.append({
                            "frame_index": frame_idx,
                            "timestamp_seconds": float(frame_idx / fps),
                            "xy": xy.astype(float).tolist(),
                            "confidence": confidence.astype(float).tolist(),
                        })
                except Exception as exc:
                    print(f"Golfie outline warning: pose detection failed at frame {frame_idx}: {exc}")
            canvas = np.zeros_like(frame)
            person_polys = []
            
            # Draw Golfer Outline (Class 0 -> Cyan)
            if results and results[0].masks is not None:
                for mask, box in zip(results[0].masks, results[0].boxes):
                    cls = int(box.cls[0])
                    poly_coords = mask.xy[0]
                    if len(poly_coords) == 0:
                        continue
                    poly_int = np.array(poly_coords, dtype=np.int32)
                    if cls == 0:
                        draw_glowing_poly(canvas, poly_int, (255, 255, 0), thickness=2)
                        person_polys.append(poly_int)

            draw_pose_overlay(canvas, pose_results)

            # The production replay is intentionally person-first. Shaft
            # detection was both the least reliable overlay and a sizeable
            # per-frame CPU cost. Draw any trusted ball point, write the mask,
            # and skip the legacy edge/Hough pass entirely.
            if person_only:
                if frame_idx in ball_lookup:
                    x, y, r = ball_lookup[frame_idx]
                    draw_glowing_circle(canvas, (int(x), int(y)), int(r), (0, 255, 0))
                elif results and results[0].masks is not None:
                    for mask, box in zip(results[0].masks, results[0].boxes):
                        if int(box.cls[0]) != 32:
                            continue
                        poly_coords = mask.xy[0]
                        if len(poly_coords) > 0:
                            draw_glowing_poly(canvas, np.array(poly_coords, dtype=np.int32), (0, 255, 0), thickness=2)
                            break
                out_writer.write(canvas)
                previous_canvas = canvas.copy()
                continue

            # Club shaft tracking. Golf clubs are not a COCO segmentation
            # class, so combine long thin edges with temporal motion and use
            # the golfer mask only as an optional hand-location prior. This
            # remains usable when a down-the-line crop omits much of the body.
            club_line = None
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                edges = cv2.Canny(blurred, 35, 110)
                motion_mask = np.zeros_like(edges)
                if previous_gray is not None:
                    delta = cv2.absdiff(blurred, previous_gray)
                    motion_mask = cv2.dilate(
                        np.where(delta > 10, 255, 0).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                    )

                person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                hands = None
                search_xmin, search_ymin, search_xmax, search_ymax = 0, 0, width, height
                if person_polys:
                    cv2.fillPoly(person_mask, person_polys, 255)
                    ys, xs = np.where(person_mask > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        ymin, ymax = int(np.min(ys)), int(np.max(ys))
                        xmin, xmax = int(np.min(xs)), int(np.max(xs))
                        g_h, g_w = max(ymax - ymin, 1), max(xmax - xmin, 1)
                        hands = np.array([(xmin + xmax) * 0.5, ymin + 0.58 * g_h])
                        # If segmentation touches an image boundary the golfer
                        # is cropped; search globally instead of trusting an
                        # incomplete body box.
                        cropped = xmin < 8 or ymin < 8 or xmax > width - 8 or ymax > height - 8
                        if not cropped:
                            search_xmin = max(0, xmin - int(1.2 * g_w))
                            search_xmax = min(width, xmax + int(1.2 * g_w))
                            search_ymin = max(0, ymin - int(0.8 * g_h))
                            search_ymax = min(height, ymax + int(0.5 * g_h))

                # Suppress clothing/body contours while retaining edges just
                # outside the silhouette where hands and shaft usually meet.
                body_suppression = cv2.dilate(
                    person_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                )
                edges[body_suppression > 0] = 0
                roi_edges = np.zeros_like(edges)
                roi_edges[search_ymin:search_ymax, search_xmin:search_xmax] = edges[
                    search_ymin:search_ymax, search_xmin:search_xmax
                ]
                lines = cv2.HoughLinesP(
                    roi_edges, rho=1, theta=np.pi / 360, threshold=18,
                    minLineLength=max(24, int(min(width, height) * 0.025)), maxLineGap=35,
                )
                best_score = -1e9
                diagonal = float(np.hypot(width, height))
                if lines is not None:
                    for detected in lines:
                        x1, y1, x2, y2 = (int(v) for v in np.asarray(detected).reshape(-1)[:4])
                        vector = np.array([x2 - x1, y2 - y1], dtype=float)
                        length = float(np.linalg.norm(vector))
                        if length < 24 or length > diagonal * 0.72:
                            continue
                        line_pixels = np.zeros_like(edges)
                        cv2.line(line_pixels, (x1, y1), (x2, y2), 255, thickness=7)
                        support = line_pixels > 0
                        motion_ratio = float(np.mean(motion_mask[support] > 0)) if np.any(support) else 0.0
                        # A static first frame cannot distinguish a shaft from
                        # alignment sticks, ropes, or horizon edges. Establish
                        # temporal state first, then require meaningful motion.
                        if previous_gray is None or motion_ratio < 0.12:
                            continue
                        score = length / diagonal * 3.0 + motion_ratio * 10.0
                        if hands is not None:
                            unit = vector / max(length, 1e-6)
                            projection = np.clip(np.dot(hands - np.array([x1, y1]), unit), 0, length)
                            distance = float(np.linalg.norm(hands - (np.array([x1, y1]) + projection * unit)))
                            score += max(0.0, 1.2 - distance / max(height * 0.12, 1.0))
                        if previous_club_line is not None:
                            previous_mid = 0.5 * (
                                np.array(previous_club_line[:2]) + np.array(previous_club_line[2:])
                            )
                            midpoint = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
                            score += max(0.0, 1.2 - np.linalg.norm(midpoint - previous_mid) / (diagonal * 0.12))
                        if score > best_score:
                            best_score = score
                            club_line = (x1, y1, x2, y2)
                if club_line is not None:
                    # Motion blur often breaks the one-pixel shaft edge into
                    # short Hough segments. Preserve its detected orientation
                    # but extend short segments so the replay communicates the
                    # full shaft rather than showing only a club-head fragment.
                    p1 = np.array(club_line[:2], dtype=float)
                    p2 = np.array(club_line[2:], dtype=float)
                    vector = p2 - p1
                    length = float(np.linalg.norm(vector))
                    minimum_visible_length = min(width, height) * 0.20
                    if 1e-6 < length < minimum_visible_length:
                        unit = vector / length
                        midpoint = 0.5 * (p1 + p2)
                        p1 = midpoint - unit * minimum_visible_length * 0.5
                        p2 = midpoint + unit * minimum_visible_length * 0.5
                        clipped, clipped_p1, clipped_p2 = cv2.clipLine(
                            (0, 0, width, height), tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int))
                        )
                        if clipped:
                            club_line = (*clipped_p1, *clipped_p2)
                    previous_club_line = club_line
                previous_gray = blurred
            except Exception as exc:
                print(f"Golfie outline warning: club-shaft detection failed at frame {frame_idx}: {exc}")
                club_line = previous_club_line

            # Render Club Outline (Amber glowing line)
            if club_line is not None:
                x1, y1, x2, y2 = club_line
                glow_color = (0, 57, 89)  # 35% brightness of amber (0, 165, 255)
                cv2.line(canvas, (x1, y1), (x2, y2), glow_color, thickness=10, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 165, 255), thickness=4, lineType=cv2.LINE_AA)

            # Draw Ball Outline (Green glowing circle) - primary tracker coordinate mapping
            if frame_idx in ball_lookup:
                x, y, r = ball_lookup[frame_idx]
                draw_glowing_circle(canvas, (int(x), int(y)), int(r), (0, 255, 0))
            elif results and results[0].masks is not None:
                # YOLO sports ball fallback (Class 32)
                for mask, box in zip(results[0].masks, results[0].boxes):
                    cls = int(box.cls[0])
                    if cls == 32:
                        poly_coords = mask.xy[0]
                        if len(poly_coords) > 0:
                            poly_int = np.array(poly_coords, dtype=np.int32)
                            draw_glowing_poly(canvas, poly_int, (0, 255, 0), thickness=2)
                            break
            
            out_writer.write(canvas)
            previous_canvas = canvas.copy()
            
        out_writer.release()
        if original_writer is not None:
            original_writer.release()
        cap.release()
        
        # Save explicit frame-timestamp mapping JSON next to the video
        map_path = output_path.with_name(f"{output_path.stem}_frame_map.json")
        with open(map_path, "w") as f:
            json.dump({
                "fps_used": float(fps),
                "start_frame": start_frame,
                "mappings": frame_mappings
            }, f, indent=2)

        if pose_analysis_output_path is not None:
            from golfie_cv.detection.swing_analysis import analyze_pose_sequence

            analysis = analyze_pose_sequence(
                pose_observations,
                camera_role=camera_role,
                impact_frame=int(impact_frame if impact_frame is not None else start_frame),
                fps=float(fps),
                handedness=handedness,
            )
            analysis_path = Path(pose_analysis_output_path)
            analysis_path.parent.mkdir(parents=True, exist_ok=True)
            analysis_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

        # Transcode AVI to browser-ready H.264 MP4
        ffmpeg_bin = _find_ffmpeg_fallback() or "ffmpeg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        def transcode(source: Path, destination: Path) -> None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                ffmpeg_bin, "-y", "-i", str(source),
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "fast", "-crf", "23", "-movflags", "+faststart",
                str(destination),
            ]
            res = subprocess.run(cmd, capture_output=True)
            if res.returncode != 0:
                import shutil
                shutil.copy(str(source), str(destination))

        transcode(temp_avi, output_path)
        if comparison_output_path is not None:
            transcode(temp_original_avi, Path(comparison_output_path))


def render_ball_detection_replay(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    output_path: Path,
    ball_track_2d: list | None = None,
) -> None:
    """Render a dark replay with the selected 2D ball track highlighted.

    This is intentionally lightweight: it shows the ball track selected by
    the pipeline without running the slower person/club YOLO overlay pass.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ball_lookup = {}
    if ball_track_2d:
        for pt in ball_track_2d:
            f_idx = getattr(pt, "frame_index", None)
            x = getattr(pt, "x_px", None)
            y = getattr(pt, "y_px", None)
            if isinstance(pt, dict):
                f_idx = pt.get("frame_index", f_idx)
                x = pt.get("x_px", x)
                y = pt.get("y_px", y)
            if f_idx is not None and x is not None and y is not None:
                ball_lookup[int(f_idx)] = (float(x), float(y))

    def draw_highlight(img, center, radius=14):
        x, y = center
        cv2.circle(img, (x, y), radius + 10, (0, 120, 120), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), radius + 4, (0, 210, 255), thickness=3, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), max(3, radius // 3), (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_avi = Path(tmpdir) / "ball_replay.avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_writer = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        recent_points = []

        for frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # Obscure the scene while keeping enough context to recognize the shot.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dark = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            dark = cv2.convertScaleAbs(dark, alpha=0.22, beta=4)

            if frame_idx in ball_lookup:
                bx, by = ball_lookup[frame_idx]
                center = (int(round(bx)), int(round(by)))
                recent_points.append(center)
                recent_points = recent_points[-18:]

            if len(recent_points) >= 2:
                for idx in range(1, len(recent_points)):
                    alpha = idx / len(recent_points)
                    color = (0, int(140 + 115 * alpha), int(160 + 95 * alpha))
                    cv2.line(dark, recent_points[idx - 1], recent_points[idx], color, thickness=2, lineType=cv2.LINE_AA)

            if frame_idx in ball_lookup and recent_points:
                draw_highlight(dark, recent_points[-1])

            out_writer.write(dark)

        out_writer.release()
        cap.release()

        ffmpeg_bin = _find_ffmpeg_fallback() or "ffmpeg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(temp_avi),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            str(output_path),
        ]

        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            import shutil
            shutil.copy(str(temp_avi), str(output_path))
