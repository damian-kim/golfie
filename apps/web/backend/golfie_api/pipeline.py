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

    log_progress("Starting real processing pipeline", ProcessingStage.DETECTING_IMPACT)

    def read_initial_frames(path, count=20):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise PipelineError(f"Could not open video file: {path}")
        frames = []
        try:
            for _ in range(count):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
        finally:
            cap.release()
        return frames

    try:
        log_progress("Reading initial background frames for Camera A...")
        initial_frames_a = read_initial_frames(session.camera_a.video_path, 20)
        log_progress("Reading initial background frames for Camera B...")
        initial_frames_b = read_initial_frames(session.camera_b.video_path, 20)
    except Exception as e:
        log_progress(f"OOM or error reading initial frames: {e}")
        raise PipelineError(f"Error reading video frames: {e}")

    if not initial_frames_a or not initial_frames_b:
        log_progress("Error: One or both videos empty")
        raise PipelineError("One or both video files contain no readable frames.")

    fps_a = session.camera_a.fps
    fps_b = session.camera_b.fps

    log_progress("Building background models...")
    bg_a = build_background_model(initial_frames_a)
    bg_b = build_background_model(initial_frames_b)

    # Free memory immediately
    del initial_frames_a
    del initial_frames_b
    import gc
    gc.collect()

    log_progress("Background modeling complete, starting candidate detection...", ProcessingStage.DETECTING_BALL)

    # 2. Candidate Detection (streaming)
    def detect_candidates_stream(path, bg_model, label="Camera"):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise PipelineError(f"Could not open video file: {path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log_progress(f"[{label}] Processing {total_frames} frames...")
        
        candidates = []
        try:
            frame_idx = 0
            while True:
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

    try:
        candidates_a = detect_candidates_stream(session.camera_a.video_path, bg_a, "Camera A")
        candidates_b = detect_candidates_stream(session.camera_b.video_path, bg_b, "Camera B")
    except Exception as e:
        log_progress(f"Error in candidate detection loop: {e}")
        raise PipelineError(f"Error processing video frames: {e}")

    log_progress("Candidate detection complete, starting 2D tracking...", ProcessingStage.TRACKING_BALL)

    # 3. 2D Tracking
    track_a = track_ball_2d(candidates_a, fps_a)
    track_b = track_ball_2d(candidates_b, fps_b)

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

    log_progress("Starting 3D triangulation...", ProcessingStage.TRIANGULATING)

    # 4. Sync-align Camera B track to Camera A timeline
    # We prefer visual alignment based on the start of the detected ball tracks
    # because it is immune to microphone latency, audio-video lag in containers,
    # or audio correlation failures.
    if track_a and track_b:
        sync_offset_frames = float(track_a[0].frame_index - track_b[0].frame_index)
        sync_offset_sec = sync_offset_frames / fps_a
        log_progress(f"Using visual track alignment. Start A: frame {track_a[0].frame_index}, Start B: frame {track_b[0].frame_index}. Visual offset: {sync_offset_sec:.3f}s ({sync_offset_frames:.1f} frames).")
    else:
        sync_offset_sec = 0.0
        sync_offset_frames = 0.0
        if session.sync is not None:
            sync_offset_sec = session.sync.offset_seconds
            sync_offset_frames = session.sync.offset_frames

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
