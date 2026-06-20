import numpy as np

from golfie_core.coordinates import CoordinateTransformer, build_right_handed_basis
from golfie_core.schemas import CoordinateSystem


def test_right_handed_basis_is_orthonormal_and_right_handed():
    x_hat, y_hat, z_hat = build_right_handed_basis(
        target_direction=np.array([1.0, 0.0, 0.0]),
        up_direction=np.array([0.0, 0.0, 1.0]),
    )
    # Orthonormal
    for v in (x_hat, y_hat, z_hat):
        assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.isclose(np.dot(x_hat, y_hat), 0.0)
    assert np.isclose(np.dot(y_hat, z_hat), 0.0)
    assert np.isclose(np.dot(x_hat, z_hat), 0.0)
    # Right-handed: x cross y == z
    assert np.allclose(np.cross(x_hat, y_hat), z_hat, atol=1e-9)


def test_right_handed_basis_handles_non_vertical_up_vector():
    """`up` doesn't need to be perfectly vertical relative to the target
    line -- only the component orthogonal to target_direction is used."""
    x_hat, y_hat, z_hat = build_right_handed_basis(
        target_direction=np.array([1.0, 0.0, 0.0]),
        up_direction=np.array([0.3, 0.0, 1.0]),  # tilted "up"
    )
    assert np.isclose(np.dot(x_hat, z_hat), 0.0)
    assert np.allclose(np.cross(x_hat, y_hat), z_hat, atol=1e-9)


def test_identity_transform_is_a_no_op():
    ct = CoordinateTransformer.identity()
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(ct.to_world(p), p)
    assert np.allclose(ct.to_rig(p), p)


def test_transform_round_trip():
    cs = CoordinateSystem(
        origin_in_rig_frame_m=[0.5, -0.2, 0.0],
        target_direction_in_rig_frame=[0.0, 1.0, 0.0],  # target line along rig's Y
        up_direction_in_rig_frame=[0.0, 0.0, 1.0],
    )
    ct = CoordinateTransformer.from_coordinate_system(cs)

    rig_point = np.array([2.0, 5.0, 1.5])
    world_point = ct.to_world(rig_point)
    recovered = ct.to_rig(world_point)
    assert np.allclose(recovered, rig_point, atol=1e-9)


def test_transform_maps_target_direction_to_world_x_axis():
    cs = CoordinateSystem(
        origin_in_rig_frame_m=[0.0, 0.0, 0.0],
        target_direction_in_rig_frame=[0.0, 1.0, 0.0],
        up_direction_in_rig_frame=[0.0, 0.0, 1.0],
    )
    ct = CoordinateTransformer.from_coordinate_system(cs)
    # A point straight down the target line in rig coords should land
    # purely on the world +X axis.
    world_point = ct.to_world(np.array([0.0, 10.0, 0.0]))
    assert np.allclose(world_point, [10.0, 0.0, 0.0], atol=1e-9)
