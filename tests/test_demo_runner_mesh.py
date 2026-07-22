"""Tests for the demo-runner mesh cache and its binary /api/mesh endpoint."""

# pylint: disable=missing-function-docstring

from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from uncertain_feedback.demo_runner import server
from uncertain_feedback.planners.mpc.kinematics import anatomical_elbow_wrist_slots
from uncertain_feedback.utils import smpl_mesh


class _FakeSmpl:
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    parents = torch.tensor(
        [
            -1,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            9,
            9,
            12,
            13,
            14,
            16,
            17,
            18,
            19,
            20,
            21,
        ]
    )

    def eval(self) -> None:
        pass

    def __call__(self, *, body_pose, global_orient, betas, transl):
        del global_orient, betas, transl
        n = body_pose.shape[0]
        vertices = torch.zeros((n, 4, 3), dtype=torch.float32)
        vertices[:, 0, 0] = body_pose[:, 15 * 3]
        joints = (
            torch.randn(
                (24, 3), generator=torch.Generator().manual_seed(0), dtype=torch.float32
            )
            .reshape(1, 24, 3)
            .repeat(n, 1, 1)
        )
        return SimpleNamespace(vertices=vertices, joints=joints)


def _cache(monkeypatch, max_entries: int = 64) -> smpl_mesh.SmplMeshCache:
    monkeypatch.setattr(smpl_mesh, "create", lambda *args, **kwargs: _FakeSmpl())
    monkeypatch.setattr(
        smpl_mesh.SmplMeshCache,
        "_fit_demo_pose",
        lambda self, positions: (
            np.zeros(3, dtype=np.float32),
            np.zeros((23, 3), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )
    body = np.random.default_rng(1).normal(size=(22, 3))
    return smpl_mesh.SmplMeshCache(body, max_entries=max_entries)


def test_mesh_cache_generates_finite_deterministic_vertices(monkeypatch) -> None:
    cache = _cache(monkeypatch)
    poses = np.random.default_rng(2).normal(size=(2, 5, 3))
    mesh_id = cache.register(poses)

    vertices = cache.vertices(mesh_id)

    assert vertices.shape == (2, 4, 3)
    assert vertices.dtype == np.dtype("<f4")
    assert np.isfinite(vertices).all()
    assert np.array_equal(cache.vertices(mesh_id), vertices)
    assert not np.array_equal(vertices[1], vertices[0])
    assert cache.faces.tolist() == [[0, 1, 2]]


def test_hand_frame_continuous_through_forearm_sweep() -> None:
    """The mesh's hand-carrying frame tracks the forearm without fabricated roll.

    ``_generate`` maps the anatomical reparameterization onto native SMPL rows:
    the recovered upper-arm orientation drives the shoulder(16) row and the pure
    forearm hinge drives the elbow(18) row, so the hand rides on joint 18's
    world frame.  Sweeping the forearm smoothly through the near-antiparallel
    region must not induce a discontinuous roll on that frame.
    """
    # Native SMPL rest arm: upper arm +x with a small forearm bend.
    rest16 = np.array([0.0, 0.0, 0.0])
    rest18 = np.array([0.30, 0.0, 0.0])
    rest20 = rest18 + 0.25 * np.array(
        [np.cos(np.deg2rad(7.5)), 0.0, np.sin(np.deg2rad(7.5))]
    )
    collar_world = Rotation.identity()
    shoulder, elbow = np.zeros(3), np.array([0.3, 0.0, 0.0])

    prev_hand = None
    prev_f = None
    max_hand_step = 0.0
    max_f_step = 0.0
    for t in np.linspace(0.0, 1.0, 400):
        ang = np.deg2rad(20 + 150 * t)
        f = np.array([np.cos(ang), 0.0, np.sin(ang)])
        wrist = elbow + 0.25 * f
        shoulder_row, elbow_row = anatomical_elbow_wrist_slots(
            shoulder, elbow, wrist, collar_world, rest18 - rest16, rest20 - rest18
        )
        shoulder_world = collar_world * Rotation.from_rotvec(shoulder_row)
        hand_frame = shoulder_world * Rotation.from_rotvec(elbow_row)  # native joint-18
        if prev_hand is not None:
            step = (prev_hand.inv() * hand_frame).magnitude()
            f_step = np.arccos(np.clip(prev_f @ f, -1.0, 1.0))
            max_hand_step = max(max_hand_step, step)
            max_f_step = max(max_f_step, f_step)
        prev_hand = hand_frame
        prev_f = f
    assert np.degrees(max_hand_step) < 1.5
    np.testing.assert_allclose(max_hand_step, max_f_step, atol=1e-9)


def test_mesh_cache_rejects_stale_identifier(monkeypatch) -> None:
    cache = _cache(monkeypatch, max_entries=1)
    rng = np.random.default_rng(3)
    stale_id = cache.register(rng.normal(size=(1, 5, 3)))
    cache.register(rng.normal(size=(1, 5, 3)))

    try:
        cache.vertices(stale_id)
    except KeyError as exc:
        assert "Unknown or stale mesh id" in str(exc)
    else:
        raise AssertionError("stale mesh id was accepted")


def test_mesh_endpoint_returns_float32_binary(monkeypatch) -> None:
    vertices = np.arange(24, dtype="<f4").reshape(2, 4, 3)
    monkeypatch.setattr(
        server,
        "rig",
        SimpleNamespace(mesh_vertices=lambda mesh_id, frame: vertices),
    )

    response = server.app.test_client().get("/api/mesh/example")

    assert response.status_code == 200
    assert response.headers["X-Mesh-Frames"] == "2"
    assert response.headers["X-Mesh-Vertices"] == "4"
    assert np.array_equal(np.frombuffer(response.data, dtype="<f4"), vertices.ravel())


def test_mesh_endpoint_reports_stale_identifier(monkeypatch) -> None:
    def missing(mesh_id: str, frame: int | None) -> np.ndarray:
        raise KeyError(f"Unknown or stale mesh id {mesh_id!r}.")

    monkeypatch.setattr(server, "rig", SimpleNamespace(mesh_vertices=missing))

    response = server.app.test_client().get("/api/mesh/stale")

    assert response.status_code == 404
    assert "Unknown or stale mesh id" in response.get_json()["error"]
