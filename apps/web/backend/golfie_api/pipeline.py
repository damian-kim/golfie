"""v0 ("Milestone 0") processing pipeline.

This intentionally does almost nothing yet -- it confirms both videos
were uploaded and read correctly, surfaces real warnings (e.g. low fps),
and returns a ShotResult where every metric is NOT_AVAILABLE. This is
the only honest thing to ship before calibration/sync/detection/
tracking/triangulation/fitting exist (spec section 19).

scripts/process_shot.py implements the same idea as a standalone CLI
(it does not depend on this module, since it has no Session/backend
concept) -- if you change the warning logic here, consider updating it
there too.
"""

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

    import cv2
    import numpy as np
    from datetime import datetime
    from golfie_api.storage import session_store
    from golfie_cv.detection import build_background_model, detect_ball_candidates
    from golfie_cv.tracking import track_ball_2d
    from golfie_cv.triangulation import triangulate_track
    from golfie_physics.fitting import fit_initial_conditions
    from golfie_physics.models.projectile import simulate_flight
    from golfie_core.schemas import MetricValue, MetricSource, TrackedPoint2D, TrackedPoint3D
    from golfie_core.schemas.shot import ShotMetrics, ShotResult

    def log_progress(msg, stage=None):
        try:
            with open(r"E:\Golfie\golfie\progress.log", "a") as f:
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

    from golfie_cv.detection import detect_relevant_window, render_ball_detection_replay, render_stripped_outlines_video
    from golfie_cv.video import read_video_metadata, ensure_constant_frame_rate

    session_dir = session_store.session_dir(session.session_id)

    def resolve_video_path(video_path):
        from pathlib import Path
        path = Path(video_path)
        return path if path.is_absolute() else session_dir / path

    orig_path_a = resolve_video_path(session.camera_a.video_path)
    orig_path_b = resolve_video_path(session.camera_b.video_path)
    
    orig_meta_a = read_video_metadata(orig_path_a)
    orig_meta_b = read_video_metadata(orig_path_b)


    log_progress("Detecting relevant window for Camera A...")
    _, _, impact_a = detect_relevant_window(orig_path_a, orig_meta_a.fps, orig_meta_a.frame_count)
    
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
    _, _, impact_b = detect_relevant_window(orig_path_b, orig_meta_b.fps, orig_meta_b.frame_count)
    
    start_time_b = max(0.0, impact_b - 3.0)
    end_time_b = min(orig_meta_b.frame_count / orig_meta_b.fps, impact_b + 2.0)
    duration_b = orig_meta_b.frame_count / orig_meta_b.fps
    if duration_b >= 5.0:
        if start_time_b == 0.0:
            end_time_b = 5.0
        elif end_time_b == duration_b:
            start_time_b = max(0.0, duration_b - 5.0)

    cfr_path_a = session_dir / f"camera_a_shot_{int(start_time_a * 1000)}_{int(end_time_a * 1000)}_cfr.mp4"
    cfr_path_b = session_dir / f"camera_b_shot_{int(start_time_b * 1000)}_{int(end_time_b * 1000)}_cfr.mp4"

    log_progress(f"Camera A relevant window: {start_time_a:.2f}s to {end_time_a:.2f}s (impact: {impact_a:.2f}s)")
    log_progress("Transcoding Camera A shot window to constant 240 FPS CFR...")
    try:
        ensure_constant_frame_rate(
            orig_path_a,
            cfr_path_a,
            target_fps=240.0,
            start_time=start_time_a,
            duration=end_time_a - start_time_a,
            progress_callback=lambda msg: log_progress(f"[Camera A] {msg}")
        )
    except Exception as e:
        log_progress(f"Transcoding Camera A failed: {e}")
        raise PipelineError(f"CFR Transcoding failed: {e}")

    log_progress(f"Camera B relevant window: {start_time_b:.2f}s to {end_time_b:.2f}s (impact: {impact_b:.2f}s)")
    log_progress("Transcoding Camera B shot window to constant 240 FPS CFR...")
    try:
        ensure_constant_frame_rate(
            orig_path_b,
            cfr_path_b,
            target_fps=240.0,
            start_time=start_time_b,
            duration=end_time_b - start_time_b,
            progress_callback=lambda msg: log_progress(f"[Camera B] {msg}")
        )
    except Exception as e:
        log_progress(f"Transcoding Camera B failed: {e}")
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
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
        background_frames_a = read_background_frames(cfr_path_a, start_frame_a, 15, 4)
        log_progress("Reading background frames for Camera B...")
        background_frames_b = read_background_frames(cfr_path_b, start_frame_b, 15, 4)
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
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
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

    # Crop candidate detection to a tight 95-frame window around the precise impact frame for tracking
    impact_frame_a = int(round(impact_cfr_a * fps_a))
    impact_frame_b = int(round(impact_cfr_b * fps_b))
    
    tracking_start_a = max(start_frame_a, impact_frame_a - 15)
    tracking_end_a = min(end_frame_a, impact_frame_a + 80)
    
    tracking_start_b = max(start_frame_b, impact_frame_b - 15)
    tracking_end_b = min(end_frame_b, impact_frame_b + 80)

    try:
        candidates_a = detect_candidates_stream(cfr_path_a, bg_a, tracking_start_a, tracking_end_a, "Camera A")
        candidates_b = detect_candidates_stream(cfr_path_b, bg_b, tracking_start_b, tracking_end_b, "Camera B")
    except Exception as e:
        log_progress(f"Error in candidate detection loop: {e}")
        raise PipelineError(f"Error processing video frames: {e}")

    log_progress("Candidate detection complete, starting 2D tracking...", ProcessingStage.TRACKING_BALL)

    # 3. 2D Tracking
    track_a = track_ball_2d(candidates_a, fps_a)
    track_b = track_ball_2d(candidates_b, fps_b)

    # Convert relative indices back to clip indices and original video timestamps.
    track_a = [
        TrackedPoint2D(
            frame_index=pt.frame_index + tracking_start_a,
            time_seconds=start_time_a + ((pt.frame_index + tracking_start_a) / fps_a),
            x_px=pt.x_px,
            y_px=pt.y_px,
            confidence=pt.confidence,
        )
        for pt in track_a
    ]
    track_b = [
        TrackedPoint2D(
            frame_index=pt.frame_index + tracking_start_b,
            time_seconds=start_time_b + ((pt.frame_index + tracking_start_b) / fps_b),
            x_px=pt.x_px,
            y_px=pt.y_px,
            confidence=pt.confidence,
        )
        for pt in track_b
    ]




    log_progress(f"2D tracking complete. Track A: {len(track_a)} pts, Track B: {len(track_b)} pts.")

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

    log_progress("Rendering down-the-line ball detection replay...", ProcessingStage.RENDERING)
    try:
        render_ball_detection_replay(
            video_path=cfr_path_a,
            start_frame=tracking_start_a,
            end_frame=tracking_end_a,
            output_path=session_dir / "camera_a_ball_replay.mp4",
            ball_track_2d=track_a,
        )
    except Exception as e:
        log_progress(f"Failed to render ball detection replay: {e}")

    log_progress("Starting 3D triangulation...", ProcessingStage.TRIANGULATING)

    # 4. Sync-align Camera B track to Camera A timeline using audio-detected physical impact timestamps
    sync_offset_sec = impact_a - impact_b
    sync_offset_frames = sync_offset_sec * fps_a
    log_progress(f"Using audio-aligned transient sync. Impact A: {impact_a:.3f}s, Impact B: {impact_b:.3f}s. Offset: {sync_offset_sec:.3f}s ({sync_offset_frames:.1f} frames)")


    aligned_track_b = [
        TrackedPoint2D(
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
        measured_3d = triangulate_track(track_a, aligned_track_b, session.calibration)
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

    log_progress("Fitting physics model initial conditions...", ProcessingStage.FITTING_PHYSICS)

    # 6. Fit Physics Initial Conditions
    try:
        fit_res = fit_initial_conditions(measured_3d)
        log_progress(f"Physics model fit: velocity={fit_res.initial_velocity_mps} m/s, speed={np.linalg.norm(fit_res.initial_velocity_mps):.2f} m/s, confidence={fit_res.confidence:.2f}")
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

    # Map simulated points to absolute timeline
    simulated_trajectory_3d = [
        TrackedPoint3D(
            time_seconds=s.time_s + t0,
            x_m=float(s.position_m[0]),
            y_m=float(s.position_m[1]),
            z_m=float(s.position_m[2]),
            confidence=fit_res.confidence,
        )
        for s in full_flight.samples
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

    metrics = ShotMetrics(
        ball_speed_mps=MetricValue(value=speed, source=MetricSource.ESTIMATED, confidence=fit_res.confidence),
        launch_angle_deg=MetricValue(value=launch_angle, source=MetricSource.ESTIMATED, confidence=fit_res.confidence),
        horizontal_launch_deg=MetricValue(value=horizontal_launch, source=MetricSource.ESTIMATED, confidence=fit_res.confidence),
        carry_m=MetricValue(value=full_flight.carry_m, source=MetricSource.ESTIMATED, confidence=fit_res.confidence),
        total_m=MetricValue(value=full_flight.carry_m, source=MetricSource.ESTIMATED, confidence=fit_res.confidence),
        apex_m=MetricValue(value=full_flight.apex_m, source=MetricSource.ESTIMATED, confidence=fit_res.confidence),
        side_deviation_m=MetricValue(value=full_flight.side_deviation_m, source=MetricSource.ESTIMATED, confidence=fit_res.confidence),
    )

    warnings = []
    if fit_res.confidence < 0.5:
        warnings.append("Low physics fitting confidence; estimated trajectory might be inaccurate.")

    log_progress(f"Processing complete! Carry: {full_flight.carry_m:.2f} m.")

    import os
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
