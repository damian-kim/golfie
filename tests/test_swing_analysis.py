from golfie_cv.detection.swing_analysis import analyze_pose_sequence


def _observation(frame, nose_x=100.0, hip_width=100.0, elbow="straight"):
    xy = [[0.0, 0.0] for _ in range(17)]
    confidence = [0.95 for _ in range(17)]
    xy[0] = [nose_x, 50.0]
    xy[5], xy[6] = [50.0, 100.0], [150.0, 100.0]
    xy[11], xy[12] = [100.0 - hip_width / 2, 200.0], [100.0 + hip_width / 2, 200.0]
    xy[5] = [60.0, 100.0]
    xy[7] = [80.0, 130.0]
    xy[9] = [100.0, 160.0] if elbow == "straight" else [120.0, 130.0]
    return {"frame_index": frame, "xy": xy, "confidence": confidence}


def test_pose_analysis_flags_supported_face_on_traits():
    observations = [
        _observation(0),
        _observation(20),
        _observation(40),
        _observation(50, hip_width=96.0),
        _observation(70, hip_width=95.0),
        _observation(100, nose_x=165.0, elbow="bent"),
        _observation(115, nose_x=170.0, elbow="bent"),
        _observation(125, nose_x=168.0, elbow="bent"),
    ]

    analysis = analyze_pose_sequence(
        observations,
        camera_role="face_on",
        impact_frame=100,
        fps=100.0,
        handedness="right",
    )

    traits = {finding["trait"] for finding in analysis["findings"]}
    assert "Excessive head translation" in traits
    assert "Limited hip-turn indication" in traits
    assert "Lead-arm collapse / chicken-wing indication" in traits


def test_pose_analysis_is_honest_when_observations_are_sparse():
    analysis = analyze_pose_sequence(
        [_observation(0), _observation(1)],
        camera_role="face_on",
        impact_frame=1,
        fps=30.0,
    )

    assert analysis["findings"] == []
    assert any("Too few" in limitation for limitation in analysis["limitations"])
