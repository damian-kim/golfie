"""End-to-end calibration-aware shot reconstruction pipeline."""

from __future__ import annotations

from golfie_core.schemas import ProcessingStage, Session, ShotResult, build_placeholder_shot_result


class PipelineError(Exception):
    pass


def run_placeholder_processing(session: Session) -> ShotResult:
    if session.camera_a is None or session.camera_b is None:
        raise PipelineError(
            "Both camera_a and camera_b must be uploaded before processing."
        )
    return build_placeholder_shot_result(
        camera_a=session.camera_a,
        camera_b=session.camera_b,
        calibration=session.calibration,
        sync=session.sync,
    )


def run_real_processing(session: Session) -> ShotResult:
    """Run the actual computer vision and physics pipeline on the session videos."""
    if session.camera_a is None or session.camera_b is None:
        raise PipelineError("Both camera_a and camera_b must be uploaded before processing.")

    if session.calibration is None or not session.calibration.is_valid:
        return build_placeholder_shot_result(
            camera_a=session.camera_a,
            camera_b=session.camera_b,
            calibration=session.calibration,
            sync=session.sync,
        )

    if (
        session.calibration.calibration_target in {"charuco", "chessboard"}
        and session.calibration.calibration_version < 2
    ):
        raise PipelineError(
            "This calibration predates the full lens/epipolar model. "
            "Re-run camera calibration before processing shots."
        )

    import cv2
    import numpy as np
    from datetime import datetime
    from golfie_api.storage import session_store
    from golfie_cv.detection import build_background_model, detect_ball_candidates
    from golfie_cv.tracking import track_ball_2d, track_ball_stereo, track_optic_ball_stereo
    from golfie_cv.triangulation import triangulate_track
    from golfie_physics.fitting import fit_initial_conditions
    from golfie_physics.models.projectile import simulate_flight
    from golfie_core.schemas import MetricValue, MetricSource, TrackedPoint2D, TrackedPoint3D
    from golfie_core.schemas.shot import ShotMetrics, ShotResult
    session_dir = session_store.session_dir(session.session_id)

    def log_progress(msg, stage=None):
        try:
            with (session_dir / "processing.log").open("a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] Session {session.session_id}: {msg}\n")
            if stage is not None:
                session.stage = stage
                session_store.save(session)
        except Exception:
            pass

    import golfie_cv
    import sys
    log_progress(f"golfie_cv imported from: {golfie_cv.__file__}")
    log_progress(f"sys.path: {sys.path[:5]}")

    log_progress("Starting real processing pipeline", ProcessingStage.DETECTING_IMPACT)

    from golfie_cv.detection import (
        detect_impact_frame_fast,
        render_ball_detection_replay,
        render_stripped_outlines_video,
    )
    from golfie_cv.video import (
        analyze_video_timing,
        ensure_constant_frame_rate,
        read_video_metadata,
        resolve_stereo_capture_rates,
    )

    def resolve_video_path(video_path):
        from pathlib import Path
        path = Path(video_path)
        return path if path.is_absolute() else session_dir / path

    orig_path_a = resolve_video_path(session.camera_a.video_path)
    orig_path_b = resolve_video_path(session.camera_b.video_path)
    
    orig_meta_a = read_video_metadata(orig_path_a)
    orig_meta_b = read_video_metadata(orig_path_b)
    timing_a = analyze_video_timing(orig_path_a)
    timing_b = analyze_video_timing(orig_path_b)
    capture_fps_a, capture_fps_b, timing_reason = resolve_stereo_capture_rates(
        timing_a, timing_b, float(session.camera_a.fps), float(session.camera_b.fps)
    )
    slowmo_a = capture_fps_a > timing_a.nominal_fps * 1.5
    slowmo_b = capture_fps_b > timing_b.nominal_fps * 1.5
    slowmo_rendered = slowmo_a and slowmo_b
    log_progress(
        "Timing analysis Camera A: "
        f"nominal={timing_a.nominal_fps:.3f}fps, average={timing_a.average_fps:.3f}fps, "
        f"VFR={timing_a.variable_frame_rate}, device={timing_a.device_model or 'metadata stripped'}, "
        f"capture={capture_fps_a:.1f}fps."
    )
    log_progress(
        "Timing analysis Camera B: "
        f"nominal={timing_b.nominal_fps:.3f}fps, average={timing_b.average_fps:.3f}fps, "
        f"VFR={timing_b.variable_frame_rate}, device={timing_b.device_model or 'metadata stripped'}, "
        f"capture={capture_fps_b:.1f}fps. Resolution: {timing_reason or timing_b.inference_reason}."
    )
    if slowmo_a != slowmo_b:
        raise PipelineError(
            "Only one camera could be identified as high-speed slow motion. Preserve native phone "
            "metadata or provide the original capture rate for the ambiguous file."
        )
    if slowmo_rendered and abs(capture_fps_a - capture_fps_b) > 1.0:
        raise PipelineError(
            f"Slow-motion capture rates differ ({capture_fps_a:.1f} vs {capture_fps_b:.1f} fps). "
            "Stereo reconstruction requires the same original capture rate."
        )

    # Intrinsics are valid only for the exact sensor orientation/resolution
    # used during calibration. Scaling/cropping them silently corrupts every
    # epipolar constraint downstream.
    expected_a = session.calibration.camera_a_image_size
    expected_b = session.calibration.camera_b_image_size
    actual_a = (orig_meta_a.width, orig_meta_a.height)
    actual_b = (orig_meta_b.width, orig_meta_b.height)
    if expected_a is not None and tuple(expected_a) != actual_a:
        raise PipelineError(
            f"Camera A recording is {actual_a[0]}x{actual_a[1]}, but calibration "
            f"was computed at {expected_a[0]}x{expected_a[1]}. Recalibrate using the shot capture mode."
        )
    if expected_b is not None and tuple(expected_b) != actual_b:
        raise PipelineError(
            f"Camera B recording is {actual_b[0]}x{actual_b[1]}, but calibration "
            f"was computed at {expected_b[0]}x{expected_b[1]}. Recalibrate using the shot capture mode."
        )
    alignment = session.calibration.coordinate_system
    if (
        alignment is not None
        and alignment.alignment_method == "camera_a_down_the_line_assumption"
        and session.camera_a.role_hint not in {None, "down_the_line"}
    ):
        raise PipelineError(
            "This MVP calibration assumes Camera A is the upright down-the-line camera, "
            "but the uploaded role says otherwise. Swap camera roles or recalibrate."
        )


    log_progress("Detecting relevant window for Camera A...")
    impact_frame_source_a, impact_a = detect_impact_frame_fast(orig_path_a)
    
    # Calculate start/end times in the original timeline, then only transcode
    # that short shot window into CFR. This avoids converting minutes of video
    # just to process ~100 ball-tracking frames.
    start_time_a = max(0.0, impact_a - 3.0)
    end_time_a = min(orig_meta_a.frame_count / orig_meta_a.fps, impact_a + 2.0)
    duration_a = orig_meta_a.frame_count / orig_meta_a.fps
    if duration_a >= 5.0:
        if start_time_a == 0.0:
            end_time_a = 5.0
        elif end_time_a == duration_a:
            start_time_a = max(0.0, duration_a - 5.0)

    log_progress("Detecting relevant window for Camera B...")
    impact_frame_source_b, impact_b = detect_impact_frame_fast(orig_path_b)
    
    start_time_b = max(0.0, impact_b - 3.0)
    end_time_b = min(orig_meta_b.frame_count / orig_meta_b.fps, impact_b + 2.0)
    duration_b = orig_meta_b.frame_count / orig_meta_b.fps
    if duration_b >= 5.0:
        if start_time_b == 0.0:
            end_time_b = 5.0
        elif end_time_b == duration_b:
            start_time_b = max(0.0, duration_b - 5.0)

    if slowmo_rendered:
        # A rendered slow-motion file stores each high-speed captured frame
        # once but assigns slow presentation timestamps. Re-encoding it at
        # 240 fps would duplicate/drop frames. Process frame ordinals from the
            # originals and apply the inferred physical capture clock only to physics.
        cfr_path_a, cfr_path_b = orig_path_a, orig_path_b
        meta_a, meta_b = orig_meta_a, orig_meta_b
        start_time_a = start_time_b = 0.0
        log_progress(
            "Rendered slow-motion mode: preserving original frame ordinals without CFR transcoding. "
            f"Container playback rates are {orig_meta_a.fps:.3f}/{orig_meta_b.fps:.3f} fps; "
            f"physical capture rates are {capture_fps_a:.1f}/{capture_fps_b:.1f} fps."
        )
    else:
        cfr_path_a = session_dir / f"camera_a_shot_{int(start_time_a * 1000)}_{int(end_time_a * 1000)}_cfr.mp4"
        cfr_path_b = session_dir / f"camera_b_shot_{int(start_time_b * 1000)}_{int(end_time_b * 1000)}_cfr.mp4"

        # Never invent temporal resolution for normal-speed recordings.
        processing_fps = float(min(orig_meta_a.fps, orig_meta_b.fps))
        capture_fps_a = capture_fps_b = processing_fps
        log_progress(
            f"Source frame rates: Camera A={orig_meta_a.fps:.3f} fps, "
            f"Camera B={orig_meta_b.fps:.3f} fps; shared processing rate={processing_fps:.3f} fps."
        )
        if processing_fps < 60.0:
            log_progress(
                "Capture-rate warning: at least one view is below 60 fps. A golf ball can travel "
                "several metres between frames, so launch reconstruction may be unavailable."
            )

        for label, source, output, start, end in (
            ("Camera A", orig_path_a, cfr_path_a, start_time_a, end_time_a),
            ("Camera B", orig_path_b, cfr_path_b, start_time_b, end_time_b),
        ):
            log_progress(f"Transcoding {label} shot window to constant {processing_fps:.3f} FPS CFR...")
            try:
                ensure_constant_frame_rate(
                    source, output, target_fps=processing_fps, start_time=start,
                    duration=end - start,
                    progress_callback=lambda msg, camera=label: log_progress(f"[{camera}] {msg}")
                )
            except Exception as e:
                log_progress(f"Transcoding {label} failed: {e}")
                raise PipelineError(f"CFR Transcoding failed: {e}")
        meta_a = read_video_metadata(cfr_path_a)
        meta_b = read_video_metadata(cfr_path_b)

    start_frame_a = 0
    end_frame_a = meta_a.frame_count
    start_frame_b = 0
    end_frame_b = meta_b.frame_count
    impact_cfr_a = impact_a - start_time_a
    impact_cfr_b = impact_b - start_time_b
    log_progress(f"Camera A CFR clip frames: {start_frame_a} to {end_frame_a} (impact: {impact_cfr_a:.2f}s in clip)")
    log_progress(f"Camera B CFR clip frames: {start_frame_b} to {end_frame_b} (impact: {impact_cfr_b:.2f}s in clip)")


    def read_background_frames(path, start_frame, count=15, step=4):
        cap = cv2.VideoCapture(str(path))
        opened = cap.isOpened()
        if not opened:
            cap.release()
            raise PipelineError(f"Could not open video file: {path}")
        frames = []
        try:
            for _ in range(start_frame):
                if not cap.grab():
                    break
            for _ in range(count):
                for _ in range(step):
                    cap.grab()
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
        finally:
            cap.release()
        return frames

    try:
        log_progress("Reading background frames for Camera A...")
        background_impact_a = (
            impact_frame_source_a if slowmo_rendered else int(round(impact_cfr_a * meta_a.fps))
        )
        background_start_a = max(start_frame_a, background_impact_a - 34)
        background_frames_a = read_background_frames(cfr_path_a, background_start_a, 12, 2)
        log_progress("Reading background frames for Camera B...")
        background_impact_b = (
            impact_frame_source_b if slowmo_rendered else int(round(impact_cfr_b * meta_b.fps))
        )
        background_start_b = max(start_frame_b, background_impact_b - 34)
        background_frames_b = read_background_frames(cfr_path_b, background_start_b, 12, 2)
    except Exception as e:
        log_progress(f"OOM or error reading background frames: {e}")
        raise PipelineError(f"Error reading video frames: {e}")

    if not background_frames_a or not background_frames_b:
        log_progress("Error: One or both background frame sets empty")
        raise PipelineError("One or both video files contain no readable background frames.")

    fps_a = meta_a.fps
    fps_b = meta_b.fps

    log_progress("Building background models...")
    bg_a = build_background_model(background_frames_a)
    bg_b = build_background_model(background_frames_b)

    # Free memory immediately
    del background_frames_a
    del background_frames_b
    import gc
    gc.collect()

    log_progress("Background modeling complete, starting candidate detection...", ProcessingStage.DETECTING_BALL)

    # 2. Candidate Detection (streaming, cropped to relevant window)
    def detect_candidates_stream(path, bg_model, start_frame, end_frame, label="Camera"):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise PipelineError(f"Could not open video file: {path}")
        
        total_frames = end_frame - start_frame
        log_progress(f"[{label}] Processing {total_frames} frames (from frame {start_frame} to {end_frame})...")
        
        candidates = []
        try:
            # Sequential decode is ordinal-accurate for iPhone B-frame/VFR MOV;
            # CAP_PROP_POS_FRAMES seeking returned frames roughly two samples
            # late in the supplied face-on clip.
            for _ in range(start_frame):
                if not cap.grab():
                    break
            frame_idx = 0
            while frame_idx < total_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                candidates.append(detect_ball_candidates(frame, bg_model))
                frame_idx += 1
                if frame_idx % 100 == 0:
                    log_progress(f"[{label}] Processed {frame_idx}/{total_frames} frames ({(frame_idx / total_frames * 100):.1f}%)")
        finally:
            cap.release()
        log_progress(f"[{label}] Finished processing {frame_idx} frames.")
        return candidates

    # Process a fixed amount of real time around impact, independent of fps.
    if slowmo_rendered:
        impact_frame_a = impact_frame_source_a
        impact_frame_b = impact_frame_source_b
    else:
        impact_frame_a = int(round(impact_cfr_a * fps_a))
        impact_frame_b = int(round(impact_cfr_b * fps_b))
    
    pre_frames_a = max(4, int(round(0.06 * capture_fps_a)))
    post_frames_a = max(12, int(round(0.25 * capture_fps_a)))
    tracking_start_a = max(start_frame_a, impact_frame_a - pre_frames_a)
    tracking_end_a = min(end_frame_a, impact_frame_a + post_frames_a)
    
    pre_frames_b = max(4, int(round(0.06 * capture_fps_b)))
    post_frames_b = max(12, int(round(0.25 * capture_fps_b)))
    tracking_start_b = max(start_frame_b, impact_frame_b - pre_frames_b)
    tracking_end_b = min(end_frame_b, impact_frame_b + post_frames_b)

    try:
        candidates_a = detect_candidates_stream(cfr_path_a, bg_a, tracking_start_a, tracking_end_a, "Camera A")
        candidates_b = detect_candidates_stream(cfr_path_b, bg_b, tracking_start_b, tracking_end_b, "Camera B")
    except Exception as e:
        log_progress(f"Error in candidate detection loop: {e}")
        raise PipelineError(f"Error processing video frames: {e}")

    log_progress("Candidate detection complete, starting 2D tracking...", ProcessingStage.TRACKING_BALL)
    for label, frames in (("Camera A", candidates_a), ("Camera B", candidates_b)):
        counts = np.asarray([len(frame) for frame in frames], dtype=np.int32)
        log_progress(
            f"[{label}] candidate diagnostics: frames={len(frames)}, total={int(counts.sum())}, "
            f"median/frame={float(np.median(counts)):.1f}, max/frame={int(counts.max()) if len(counts) else 0}."
        )

    # 3. 2D Tracking
    if session.calibration.calibration_version >= 2:
        track_a, track_b = track_optic_ball_stereo(
            candidates_a, candidates_b, 0.5 * (capture_fps_a + capture_fps_b), session.calibration
        )
        tracking_method = "optic-color physics-scored"
        if not track_a or not track_b:
            track_a, track_b = track_ball_stereo(
                candidates_a, candidates_b,
                0.5 * (capture_fps_a + capture_fps_b), session.calibration,
                max_temporal_shift_frames=4,
            )
            tracking_method = "generic stereo fallback"
    else:
        track_a, track_b = [], []
        tracking_method = "unavailable"
    if track_a and track_b:
        local_sync_shift = track_a[0].frame_index - track_b[0].frame_index
        log_progress(
            f"{tracking_method} tracking selected {len(track_a)} geometry-consistent paired detections; "
            f"residual temporal shift={local_sync_shift:+d} captured frames "
            f"({1000.0 * local_sync_shift / max(capture_fps_a, 1.0):+.2f}ms)."
        )
    else:
        log_progress(
            "Stereo-aware tracking found no joint hypothesis; trying independent tracks for diagnostics."
        )
        track_a = track_ball_2d(candidates_a, fps_a)
        track_b = track_ball_2d(candidates_b, fps_b)

    # Stereo selection already establishes correspondence. In rendered
    # slow-motion mode, build a shared physical timeline from each selected
    # track's first captured frame so residual impact-detector offsets do not
    # leak back into triangulation.
    local_origin_a = track_a[0].frame_index if track_a else 0
    local_origin_b = track_b[0].frame_index if track_b else 0
    track_a = [
        TrackedPoint2D(
            frame_index=pt.frame_index + tracking_start_a,
            time_seconds=(
                (pt.frame_index - local_origin_a) / capture_fps_a
                if slowmo_rendered else
                start_time_a + ((pt.frame_index + tracking_start_a) / fps_a)
            ),
            x_px=pt.x_px,
            y_px=pt.y_px,
            confidence=pt.confidence,
        )
        for pt in track_a
    ]
    track_b = [
        TrackedPoint2D(
            frame_index=pt.frame_index + tracking_start_b,
            time_seconds=(
                (pt.frame_index - local_origin_b) / capture_fps_b
                if slowmo_rendered else
                start_time_b + ((pt.frame_index + tracking_start_b) / fps_b)
            ),
            x_px=pt.x_px,
            y_px=pt.y_px,
            confidence=pt.confidence,
        )
        for pt in track_b
    ]




    def track_diagnostics(track):
        positions = np.asarray([(p.x_px, p.y_px) for p in track], dtype=np.float64)
        if len(positions) == 0:
            return "points=0"
        unique = len({(round(float(x), 1), round(float(y), 1)) for x, y in positions})
        steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        return (
            f"points={len(track)}, unique_positions={unique} ({unique / len(track):.0%}), "
            f"displacement={np.linalg.norm(positions[-1] - positions[0]):.1f}px, "
            f"median_step={float(np.median(steps)) if len(steps) else 0.0:.1f}px"
        )

    log_progress(f"2D tracking complete. Camera A: {track_diagnostics(track_a)}.")
    log_progress(f"2D tracking complete. Camera B: {track_diagnostics(track_b)}.")

    if not track_a or not track_b:
        log_progress("Tracking failed to resolve ball tracks in both cameras.")
        return ShotResult(
            metrics=ShotMetrics(),
            measured_points_3d=[],
            fitted_points_3d=[],
            simulated_trajectory_3d=[],
            warnings=["No 2D ball tracks could be resolved in one or both camera views."],
            is_placeholder=False,
            notes="Real pipeline ran but failed to find/track the ball."
        )

    import os
    if os.getenv("GOLFIE_RENDER_BALL_REPLAYS", "").lower() in {"1", "true", "yes"}:
        log_progress("Rendering ball-selection diagnostic replays...", ProcessingStage.RENDERING)
        for camera_name, video_path, start_frame, end_frame, track in (
            ("camera_a", cfr_path_a, tracking_start_a, tracking_end_a, track_a),
            ("camera_b", cfr_path_b, tracking_start_b, tracking_end_b, track_b),
        ):
            try:
                render_ball_detection_replay(
                    video_path=video_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    output_path=session_dir / f"{camera_name}_ball_replay.mp4",
                    ball_track_2d=track,
                )
                log_progress(f"Rendered {camera_name} ball-selection replay.")
            except Exception as e:
                log_progress(f"Failed to render {camera_name} ball-selection replay: {type(e).__name__}: {e}")
    else:
        log_progress(
            "Skipping optional ball-selection replay render for latency "
            "(set GOLFIE_RENDER_BALL_REPLAYS=1 to enable)."
        )

    log_progress("Starting 3D triangulation...", ProcessingStage.TRIANGULATING)

    # 4. Sync-align Camera B track to Camera A timeline using audio-detected physical impact timestamps
    if slowmo_rendered:
        sync_offset_sec = 0.0
        sync_source = "impact-relative slow-motion frame alignment"
    elif session.sync is not None and session.sync.confidence >= 0.2:
        sync_offset_sec = float(session.sync.offset_seconds)
        sync_source = f"{session.sync.method.value} sync"
    else:
        sync_offset_sec = impact_a - impact_b
        sync_source = "independently detected impact fallback"
    sync_offset_frames = sync_offset_sec * fps_a
    log_progress(
        f"Using {sync_source}. Impact A: {impact_a:.3f}s, Impact B: {impact_b:.3f}s. "
        f"Offset: {sync_offset_sec:.6f}s ({sync_offset_frames:.2f} frames)"
    )


    aligned_track_b = [
        TrackedPoint2D(
            # frame_index remains diagnostic only; time_seconds is the source
            # of truth because offsets are generally fractional frames.
            frame_index=pt.frame_index + int(round(sync_offset_frames)),
            time_seconds=pt.time_seconds + sync_offset_sec,
            x_px=pt.x_px,
            y_px=pt.y_px,
            confidence=pt.confidence,
        )
        for pt in track_b
    ]

    # Diagnostic: log track ranges after alignment to verify temporal overlap
    if track_a and aligned_track_b:
        ta_times = [pt.time_seconds for pt in track_a]
        tb_times = [pt.time_seconds for pt in aligned_track_b]
        ta_frames = [pt.frame_index for pt in track_a]
        tb_frames = [pt.frame_index for pt in aligned_track_b]
        log_progress(
            f"Track A: frames {min(ta_frames)}-{max(ta_frames)}, "
            f"times {min(ta_times):.4f}s-{max(ta_times):.4f}s"
        )
        log_progress(
            f"Track B (aligned): frames {min(tb_frames)}-{max(tb_frames)}, "
            f"times {min(tb_times):.4f}s-{max(tb_times):.4f}s"
        )
        t_overlap_start = max(min(ta_times), min(tb_times))
        t_overlap_end = min(max(ta_times), max(tb_times))
        log_progress(f"Temporal overlap: {t_overlap_end - t_overlap_start:.4f}s")

    # 5. Triangulation
    try:
        measured_3d = triangulate_track(
            track_a,
            aligned_track_b,
            session.calibration,
            epipolar_limit_px=18.0 if tracking_method == "optic-color physics-scored" else None,
        )
        log_progress(f"Triangulated {len(measured_3d)} points in 3D.")
    except Exception as e:
        log_progress(f"Triangulation failed: {e}")
        return ShotResult(
            metrics=ShotMetrics(),
            measured_points_3d=[],
            fitted_points_3d=[],
            simulated_trajectory_3d=[],
            warnings=[f"3D Triangulation failed: {e}"],
            is_placeholder=False,
            notes="Triangulation step failed."
        )

    if len(measured_3d) < 3:
        log_progress("Too few 3D points to fit model.")
        return ShotResult(
            metrics=ShotMetrics(),
            measured_points_3d=measured_3d,
            fitted_points_3d=[],
            simulated_trajectory_3d=[],
            warnings=["Insufficient 3D triangulated points to fit trajectory (minimum 3 required)."],
            is_placeholder=False,
            notes="Triangulation produced too few points."
        )

    positions_3d = np.asarray(
        [(point.x_m, point.y_m, point.z_m) for point in measured_3d], dtype=np.float64
    )
    unique_3d = len({tuple(np.round(position, 3)) for position in positions_3d})
    step_3d = np.linalg.norm(np.diff(positions_3d, axis=0), axis=1)
    stationary_fraction = float(np.mean(step_3d < 0.005)) if len(step_3d) else 1.0
    duration_3d = float(measured_3d[-1].time_seconds - measured_3d[0].time_seconds)
    displacement_3d = float(np.linalg.norm(positions_3d[-1] - positions_3d[0]))
    reprojection = [
        float(point.reprojection_error_px)
        for point in measured_3d
        if point.reprojection_error_px is not None
    ]
    log_progress(
        "3D track diagnostics: "
        f"points={len(measured_3d)}, unique@1mm={unique_3d} ({unique_3d / len(measured_3d):.0%}), "
        f"duration={duration_3d:.4f}s, displacement={displacement_3d:.3f}m, "
        f"stationary_steps(<5mm)={stationary_fraction:.0%}, "
        f"median_reprojection={float(np.median(reprojection)) if reprojection else float('nan'):.2f}px."
    )
    rejection_reasons = []
    if unique_3d < 4 or unique_3d / len(measured_3d) < 0.5:
        rejection_reasons.append("too many repeated 3D positions")
    if stationary_fraction > 0.4:
        rejection_reasons.append("too many stationary consecutive samples")
    if duration_3d <= 0.0 or displacement_3d < 0.05:
        rejection_reasons.append("insufficient measured flight displacement")
    if rejection_reasons:
        reason = "; ".join(rejection_reasons)
        log_progress(f"Rejecting false ball track before physics fitting: {reason}.")
        return ShotResult(
            metrics=ShotMetrics(),
            measured_points_3d=measured_3d,
            fitted_points_3d=[],
            simulated_trajectory_3d=[],
            warnings=[
                "Ball-flight reconstruction was rejected because the selected stereo track is not "
                f"consistent with a moving golf ball ({reason}). Check the ball-selection debug replays."
            ],
            is_placeholder=False,
            notes="Stereo correspondences existed, but trajectory quality gates rejected a false track."
        )

    # Establish the shot-local tee origin. Stereo calibration determines
    # scale/orientation but a waved board does not define the ball's location.
    origin = np.array(
        [measured_3d[0].x_m, measured_3d[0].y_m, measured_3d[0].z_m], dtype=np.float64
    )
    measured_3d = [
        TrackedPoint3D(
            time_seconds=point.time_seconds,
            x_m=float(point.x_m - origin[0]),
            y_m=float(point.y_m - origin[1]),
            z_m=float(point.z_m - origin[2]),
            confidence=point.confidence,
            reprojection_error_px=point.reprojection_error_px,
        )
        for point in measured_3d
    ]

    log_progress("Fitting physics model initial conditions...", ProcessingStage.FITTING_PHYSICS)

    # 6. Fit Physics Initial Conditions
    try:
        fit_res = fit_initial_conditions(
            measured_3d, time_origin_seconds=measured_3d[0].time_seconds
        )
        speed = float(np.linalg.norm(fit_res.initial_velocity_mps))
        log_progress(
            f"Physics fit diagnostics: velocity={fit_res.initial_velocity_mps} m/s, "
            f"speed={speed:.2f}m/s, residual_rms={fit_res.residual_rms_m:.4f}m, "
            f"confidence={fit_res.confidence:.3f}, optimizer_success={fit_res.optimizer_success}, "
            f"active_bounds={fit_res.active_bounds or 'none'}, message={fit_res.optimizer_message!r}."
        )
    except Exception as e:
        log_progress(f"Fitting failed: {e}")
        return ShotResult(
            metrics=ShotMetrics(),
            measured_points_3d=measured_3d,
            fitted_points_3d=[],
            simulated_trajectory_3d=[],
            warnings=[f"Initial conditions fitting failed: {e}"],
            is_placeholder=False,
            notes="Physics model fitting step failed."
        )

    vertical_velocity = float(fit_res.initial_velocity_mps[2])
    if (
        not fit_res.optimizer_success
        or fit_res.active_bounds
        or fit_res.confidence < 0.2
        or speed < 5.0
        or speed > 95.0
        or vertical_velocity <= 0.0
    ):
        reasons = []
        if not fit_res.optimizer_success:
            reasons.append("optimizer did not converge")
        if fit_res.active_bounds:
            reasons.append(f"parameters hit bounds: {', '.join(fit_res.active_bounds)}")
        if fit_res.confidence < 0.2:
            reasons.append(f"fit confidence {fit_res.confidence:.2f} is below 0.20")
        if speed < 5.0 or speed > 95.0:
            reasons.append(f"launch speed {speed:.1f} m/s is outside the supported range")
        if vertical_velocity <= 0.0:
            reasons.append(f"vertical launch velocity {vertical_velocity:.1f} m/s is not upward")
        reason = "; ".join(reasons)
        log_progress(f"Rejecting physics fit and withholding metrics: {reason}.")
        return ShotResult(
            metrics=ShotMetrics(),
            measured_points_3d=measured_3d,
            fitted_points_3d=[],
            simulated_trajectory_3d=[],
            warnings=[f"Trajectory metrics are unavailable because the physics fit was nonphysical ({reason})."],
            is_placeholder=False,
            notes="Triangulation completed, but physics validation rejected the trajectory."
        )

    log_progress("Simulating full flight trajectory...", ProcessingStage.RENDERING)

    # 7. Simulate Full Flight (RK4 Solver)
    t0 = measured_3d[0].time_seconds
    full_flight = simulate_flight(
        initial_position_m=fit_res.initial_position_m,
        initial_velocity_mps=fit_res.initial_velocity_mps,
        params=fit_res.params,
        max_time_s=12.0,
        dt=0.001
    )

    # Map a bounded number of simulated points to the absolute timeline.  The
    # solver keeps its 1 ms integration step for stable landing metrics, but
    # sending several thousand nearly-collinear points only bloats session JSON
    # and makes the browser do unnecessary work.
    flight_confidence = min(float(fit_res.confidence), 0.65)
    flight_samples = full_flight.samples
    if len(flight_samples) > 400:
        sample_indices = np.linspace(0, len(flight_samples) - 1, 400).round().astype(int)
        flight_samples = [flight_samples[int(index)] for index in sample_indices]
    simulated_trajectory_3d = [
        TrackedPoint3D(
            time_seconds=s.time_s + t0,
            x_m=float(s.position_m[0]),
            y_m=float(s.position_m[1]),
            z_m=float(s.position_m[2]),
            confidence=flight_confidence,
        )
        for s in flight_samples
    ]

    # Map fitted points (sampled at measured timestamps)
    fitted_flight = simulate_flight(
        initial_position_m=fit_res.initial_position_m,
        initial_velocity_mps=fit_res.initial_velocity_mps,
        params=fit_res.params,
        max_time_s=float(measured_3d[-1].time_seconds - t0 + 0.01),
        dt=0.001
    )
    sim_times = np.array([s.time_s for s in fitted_flight.samples], dtype=np.float64)
    sim_pos = np.array([s.position_m for s in fitted_flight.samples], dtype=np.float64)

    t_rel = np.array([p.time_seconds - t0 for p in measured_3d], dtype=np.float64)
    fitted_points_3d = []
    for idx, p_meas in enumerate(measured_3d):
        int_x = float(np.interp(t_rel[idx], sim_times, sim_pos[:, 0]))
        int_y = float(np.interp(t_rel[idx], sim_times, sim_pos[:, 1]))
        int_z = float(np.interp(t_rel[idx], sim_times, sim_pos[:, 2]))
        fitted_points_3d.append(
            TrackedPoint3D(
                time_seconds=p_meas.time_seconds,
                x_m=int_x,
                y_m=int_y,
                z_m=int_z,
                confidence=fit_res.confidence,
            )
        )

    # 8. Reconstruct Shot Metrics
    speed = float(np.linalg.norm(fit_res.initial_velocity_mps))
    launch_angle = float(np.degrees(np.arcsin(fit_res.initial_velocity_mps[2] / speed)))
    horizontal_launch = float(np.degrees(np.arctan2(fit_res.initial_velocity_mps[1], fit_res.initial_velocity_mps[0])))

    launch_notes = "Estimated from the first stereo-reconstructed ball observations."
    direction_notes = (
        "Target line is inferred from the Camera A down-the-line orientation; "
        "use an explicitly aligned rig for simulator-grade left/right values."
    )
    flight_notes = (
        "Drag-only flight estimate to first ground contact; spin/lift and bounce/roll "
        "are not measured by this MVP."
    )
    metrics = ShotMetrics(
        ball_speed_mps=MetricValue(
            value=speed,
            source=MetricSource.ESTIMATED,
            confidence=fit_res.confidence,
            notes=launch_notes,
        ),
        launch_angle_deg=MetricValue(
            value=launch_angle,
            source=MetricSource.ESTIMATED,
            confidence=fit_res.confidence,
            notes=launch_notes,
        ),
        horizontal_launch_deg=MetricValue(
            value=horizontal_launch,
            source=MetricSource.ESTIMATED,
            confidence=flight_confidence,
            notes=direction_notes,
        ),
        carry_m=MetricValue(
            value=full_flight.carry_m,
            source=MetricSource.ESTIMATED,
            confidence=flight_confidence,
            notes=flight_notes,
        ),
        total_m=MetricValue(
            value=full_flight.carry_m,
            source=MetricSource.EXPERIMENTAL,
            confidence=min(flight_confidence, 0.35),
            notes="Reported at first ground contact because bounce and roll are not modeled.",
        ),
        apex_m=MetricValue(
            value=full_flight.apex_m,
            source=MetricSource.ESTIMATED,
            confidence=flight_confidence,
            notes=flight_notes,
        ),
        side_deviation_m=MetricValue(
            value=full_flight.side_deviation_m,
            source=MetricSource.ESTIMATED,
            confidence=flight_confidence,
            notes=f"{flight_notes} {direction_notes}",
        ),
    )

    warnings = []
    if fit_res.confidence < 0.5:
        warnings.append("Low physics fitting confidence; estimated trajectory might be inaccurate.")

    log_progress(f"Processing complete! Carry: {full_flight.carry_m:.2f} m.")

    if os.getenv("GOLFIE_RENDER_OUTLINES", "").lower() in {"1", "true", "yes"}:
        log_progress("Rendering stripped outlines video for Camera A...", ProcessingStage.RENDERING)
        session_dir = session_store.session_dir(session.session_id)
        try:
            render_stripped_outlines_video(
                video_path=cfr_path_a,
                start_frame=start_frame_a,
                end_frame=end_frame_a,
                output_path=session_dir / "camera_a_stripped.mp4",
                background_model=bg_a,
                ball_track_2d=track_a
            )
        except Exception as e:
            log_progress(f"Failed to render outlines video for Camera A: {e}")

        log_progress("Rendering stripped outlines video for Camera B...")
        try:
            render_stripped_outlines_video(
                video_path=cfr_path_b,
                start_frame=start_frame_b,
                end_frame=end_frame_b,
                output_path=session_dir / "camera_b_stripped.mp4",
                background_model=bg_b,
                ball_track_2d=track_b
            )
        except Exception as e:
            log_progress(f"Failed to render outlines video for Camera B: {e}")
    else:
        log_progress("Skipping stripped outline debug render (set GOLFIE_RENDER_OUTLINES=1 to enable).", ProcessingStage.RENDERING)

    return ShotResult(
        metrics=metrics,
        measured_points_3d=measured_3d,
        fitted_points_3d=fitted_points_3d,
        simulated_trajectory_3d=simulated_trajectory_3d,
        warnings=warnings,
        is_placeholder=False,
        notes="Shot reconstructed and simulated successfully via the end-to-end computer vision and physics pipeline."
    )


def advance_through_placeholder_stages(session: Session) -> Session:
    """Synchronously 'run' every pipeline stage.

    Runs real image-processing and trajectory-fitting steps if valid
    calibration data is present.
    """
    from golfie_api.storage import session_store

    session.stage = ProcessingStage.EXTRACTING_FRAMES
    session_store.save(session)

    session.stage = ProcessingStage.SYNCING
    session_store.save(session)
    
    # Run real video synchronization
    try:
        from golfie_cv.sync import estimate_sync_offset
        session.sync = estimate_sync_offset(session.camera_a.video_path, session.camera_b.video_path)
    except Exception as e:
        from golfie_core.schemas import SyncResult, SyncMethod
        session.sync = SyncResult(
            offset_seconds=0.0,
            offset_frames=0.0,
            method=SyncMethod.MANUAL,
            confidence=0.0,
            notes=f"Auto audio sync failed: {e}",
        )
    session_store.save(session)

    if session.calibration is not None and session.calibration.is_valid:
        session.shot = run_real_processing(session)
    else:
        # Quickly cycle stages for placeholder/simulated data
        session.stage = ProcessingStage.DETECTING_IMPACT
        session_store.save(session)
        
        # Detect relevant window and render outlines if videos are available
        if session.camera_a and session.camera_b:
            import cv2
            import numpy as np
            from golfie_cv.detection import detect_relevant_window, render_stripped_outlines_video
            from golfie_cv.video import read_video_metadata

            try:
                session_dir = session_store.session_dir(session.session_id)
                cfr_path_a = session_dir / "camera_a_cfr.mp4"
                cfr_path_b = session_dir / "camera_b_cfr.mp4"

                from golfie_cv.video import ensure_constant_frame_rate
                ensure_constant_frame_rate(session.camera_a.video_path, cfr_path_a, target_fps=240.0)
                ensure_constant_frame_rate(session.camera_b.video_path, cfr_path_b, target_fps=240.0)

                meta_a = read_video_metadata(cfr_path_a)
                start_a, end_a, _ = detect_relevant_window(cfr_path_a, meta_a.fps, meta_a.frame_count)
                
                meta_b = read_video_metadata(cfr_path_b)
                start_b, end_b, _ = detect_relevant_window(cfr_path_b, meta_b.fps, meta_b.frame_count)
                
                # Build background models
                cap_a = cv2.VideoCapture(str(cfr_path_a))
                bg_a = None
                if cap_a.isOpened():
                    frames = []
                    for _ in range(20):
                        ok, f = cap_a.read()
                        if ok: frames.append(f)
                    cap_a.release()
                    if frames:
                        bg_a = np.median(frames, axis=0).astype(np.uint8)
                        
                cap_b = cv2.VideoCapture(str(cfr_path_b))
                bg_b = None
                if cap_b.isOpened():
                    frames = []
                    for _ in range(20):
                        ok, f = cap_b.read()
                        if ok: frames.append(f)
                    cap_b.release()
                    if frames:
                        bg_b = np.median(frames, axis=0).astype(np.uint8)

                session.stage = ProcessingStage.RENDERING
                session_store.save(session)

                render_stripped_outlines_video(
                    video_path=cfr_path_a,
                    start_frame=start_a,
                    end_frame=end_a,
                    output_path=session_dir / "camera_a_stripped.mp4",
                    background_model=bg_a,
                    ball_track_2d=None
                )
                render_stripped_outlines_video(
                    video_path=cfr_path_b,
                    start_frame=start_b,
                    end_frame=end_b,
                    output_path=session_dir / "camera_b_stripped.mp4",
                    background_model=bg_b,
                    ball_track_2d=None
                )
            except Exception:
                pass

        session.stage = ProcessingStage.DETECTING_BALL
        session_store.save(session)
        session.stage = ProcessingStage.TRACKING_BALL
        session_store.save(session)
        session.stage = ProcessingStage.TRIANGULATING
        session_store.save(session)
        session.stage = ProcessingStage.FITTING_PHYSICS
        session_store.save(session)
        session.stage = ProcessingStage.RENDERING
        session_store.save(session)
        session.shot = run_placeholder_processing(session)

    session.stage = ProcessingStage.DONE
    session_store.save(session)
    return session
