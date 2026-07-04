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
    ball_track_2d: list | None = None
) -> None:
    """Run YOLOv8 segmentation on the relevant frame range of the video.
    
    Draws glowing neon-colored outlines of the person (cyan), golf club (amber), and ball (green)
    on a black background. Detects the club using a deterministic Canny + Hough Line solver.
    Transcodes the final output to H.264 MP4 and logs the frame-timestamp mapping JSON.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    # 1. Load YOLOv8 segmentation model
    from ultralytics import YOLO
    model = YOLO("yolov8n-seg.pt")
    
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

    frame_mappings = []

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_avi = Path(tmpdir) / "temp_outlines.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_writer = cv2.VideoWriter(str(temp_avi), fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
                
            # Log exact frame mapping and timestamp matching VFR standards
            cropped_idx = frame_idx - start_frame
            frame_mappings.append({
                "original_frame_idx": frame_idx,
                "cropped_frame_idx": cropped_idx,
                "original_timestamp_sec": round(frame_idx / fps, 5),
                "cropped_timestamp_sec": round(cropped_idx / fps, 5)
            })

            # Run YOLOv8 segmentation on the current frame
            results = model(frame, verbose=False)
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

            # Deterministic Club Shaft Line Tracking (Canny + Hough Transform)
            club_line = None
            if person_polys:
                try:
                    person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(person_mask, person_polys, 255)
                    
                    ys, xs = np.where(person_mask > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        ymin, ymax = int(np.min(ys)), int(np.max(ys))
                        xmin, xmax = int(np.min(xs)), int(np.max(xs))
                        g_h = ymax - ymin
                        g_w = xmax - xmin
                        
                        # Locate hands region (lower-middle section of golfer mask)
                        hands_x = int((xmin + xmax) / 2)
                        hands_y = int(ymin + 0.55 * g_h)
                        
                        # Restrict line search to golfer's swing quadrant
                        search_ymin = max(0, ymin - int(0.3 * g_h))
                        search_ymax = min(height, ymax + int(0.2 * g_h))
                        search_xmin = max(0, xmin - int(0.6 * g_w))
                        search_xmax = min(width, xmax + int(0.6 * g_w))
                        
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        edges = cv2.Canny(blurred, 50, 150)
                        
                        # Erase golfer's inner body boundaries to ignore shirts/clothing lines
                        body_dilated = cv2.dilate(person_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
                        edges[body_dilated > 0] = 0
                        
                        roi_edges = np.zeros_like(edges)
                        roi_edges[search_ymin:search_ymax, search_xmin:search_xmax] = edges[search_ymin:search_ymax, search_xmin:search_xmax]
                        
                        lines = cv2.HoughLinesP(
                            roi_edges,
                            rho=1,
                            theta=np.pi/180,
                            threshold=25,
                            minLineLength=35,
                            maxLineGap=10
                        )
                        
                        best_line = None
                        min_dist_to_hands = float("inf")
                        
                        if lines is not None:
                            for line in lines:
                                x1, y1, x2, y2 = line[0]
                                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                                if 35 < length < 250:
                                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                                    # Ignore perfectly horizontal/vertical noise line segments
                                    if 10 < angle < 85:
                                        # Distance from line segment to golfer's estimated hand location
                                        line_vec = np.array([x2 - x1, y2 - y1])
                                        point_vec = np.array([hands_x - x1, hands_y - y1])
                                        line_len = np.linalg.norm(line_vec)
                                        line_unit = line_vec / (line_len + 1e-6)
                                        proj_len = np.clip(np.dot(point_vec, line_unit), 0, line_len)
                                        closest_pt = np.array([x1, y1]) + proj_len * line_unit
                                        dist = np.linalg.norm(np.array([hands_x, hands_y]) - closest_pt)
                                        
                                        if dist < min_dist_to_hands and dist < g_h * 0.4:
                                            min_dist_to_hands = dist
                                            best_line = (x1, y1, x2, y2)
                        if best_line is not None:
                            club_line = best_line
                except Exception:
                    pass

            # Render Club Outline (Amber glowing line)
            if club_line is not None:
                x1, y1, x2, y2 = club_line
                glow_color = (0, 57, 89)  # 35% brightness of amber (0, 165, 255)
                cv2.line(canvas, (x1, y1), (x2, y2), glow_color, thickness=6, lineType=cv2.LINE_AA)
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 165, 255), thickness=2, lineType=cv2.LINE_AA)

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
            
        out_writer.release()
        cap.release()
        
        # Save explicit frame-timestamp mapping JSON next to the video
        map_path = output_path.with_name(f"{output_path.stem}_frame_map.json")
        with open(map_path, "w") as f:
            json.dump({
                "fps_used": float(fps),
                "start_frame": start_frame,
                "mappings": frame_mappings
            }, f, indent=2)

        # Transcode AVI to browser-ready H.264 MP4
        ffmpeg_bin = _find_ffmpeg_fallback() or "ffmpeg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(temp_avi),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            str(output_path)
        ]
        
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            import shutil
            shutil.copy(str(temp_avi), str(output_path))
