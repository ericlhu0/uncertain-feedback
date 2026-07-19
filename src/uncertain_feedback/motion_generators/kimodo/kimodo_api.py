"""kimodo (NVIDIA) text-to-motion backend.

:class:`KimodoMotionGenerator` implements the :class:`MotionGenerator`
interface using NVIDIA's kimodo SMPL-X diffusion model
(``github.com/nv-tlabs/kimodo``).

kimodo pins ``pydantic>=2`` and ``transformers==5.1.0``, which conflict with
this project's main environment, so generation runs in an isolated conda env
(:data:`KIMODO_CONDA_ENV`) via the standalone worker
``_kimodo_inference_worker.py`` — mirroring the SAM/MHR subprocess pattern in
``data_collection``.  Only the two ``generate_*`` methods shell out; the
pose-handling methods run in the main environment on SMPL ``body_pose (21, 3)``
arrays using the shared ``hml_smpl_conversion`` helpers.

The worker exports Kimodo local rotations as AMASS-style ``pose_body`` for
arm-angle extraction, but position samples and standalone videos use
Kimodo-native SMPL-X skeleton FK. This avoids treating Kimodo residual
rotations as this repo SMPL FK convention.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from uncertain_feedback.consts import (
    KIMODO_CONDA_ENV,
    KIMODO_MODEL,
    KIMODO_START_POSE_PATH,
)
from uncertain_feedback.motion_generators.base import MotionGenerator
from uncertain_feedback.motion_generators.mdm.hml_smpl_conversion import (
    ARM_BODY_POSE_INDICES,
    smpl_body_pose_to_arm_aa,
    smpl_body_pose_to_collar_aa,
    smpl_body_pose_to_positions,
    smpl_body_pose_to_spine3_aa,
)

_FPS = 30  # kimodo's native frame rate (model config.yaml: fps=30)


def _clean_env() -> dict[str, str]:
    """Parent env minus the caller's uv/venv vars (so the kimodo env's python
    resolves its own packages, not the launcher's uv venv)."""
    return {
        k: v
        for k, v in os.environ.items()
        if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")
    }


class KimodoMotionGenerator(MotionGenerator):
    """kimodo SMPL-X text-to-motion backend (subprocess-isolated)."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conda_env: str = KIMODO_CONDA_ENV,
        seed: int = 10,
        num_denoising_steps: int = 100,
    ) -> None:
        super().__init__()
        # ``model_path`` overrides the kimodo model name when given (the builder
        # passes the shared --model-path); otherwise use the SMPL-X default.
        self._model_name = str(model_path) if model_path is not None else KIMODO_MODEL
        self._conda_env = conda_env
        self._seed = seed  # forwarded to the worker for reproducible sampling
        self._num_denoising_steps = num_denoising_steps  # DDIM steps per sample
        self._worker_path = Path(__file__).parent / "_kimodo_inference_worker.py"
        self._python: str | None = None  # resolved kimodo env interpreter

    # ------------------------------------------------------------------
    # Pose handling (main env, SMPL body_pose (21, 3) — no kimodo / no Llama)
    # ------------------------------------------------------------------

    def load_pose(self, path: str | Path) -> np.ndarray:
        """Load a ``(21, 3)`` SMPL body_pose start pose from a ``.npy`` file.

        Falls back to :data:`KIMODO_START_POSE_PATH` when ``path`` does not
        exist, and to a zero (T-pose) body_pose if that asset is also missing.
        """
        path = Path(path)
        if path.suffix == ".pt" or not path.exists():
            path = KIMODO_START_POSE_PATH
        if not path.exists():
            return np.zeros((21, 3), dtype=np.float64)
        return np.asarray(np.load(path), dtype=np.float64).reshape(21, 3)

    def decode_pose(
        self, pose: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        body_pose = np.asarray(pose, dtype=np.float64).reshape(21, 3)
        arm_aa = smpl_body_pose_to_arm_aa(body_pose)
        body_positions = smpl_body_pose_to_positions(
            body_pose, self._fk.tpose_all_joints
        )
        spine3_aa = smpl_body_pose_to_spine3_aa(body_pose)
        collar_aa = smpl_body_pose_to_collar_aa(body_pose)
        return arm_aa, body_positions, spine3_aa, collar_aa

    def build_pose_from_arm_aa(
        self,
        base_pose: np.ndarray,
        arm_aa: np.ndarray,
    ) -> np.ndarray:
        pose = np.asarray(base_pose, dtype=np.float64).reshape(21, 3).copy()
        pose[ARM_BODY_POSE_INDICES, :] = np.asarray(arm_aa, dtype=np.float64)
        return pose

    # ------------------------------------------------------------------
    # Generation (isolated conda env via subprocess worker)
    # ------------------------------------------------------------------

    def generate_left_arm_trajectory(
        self,
        text: str,
        motion_length_seconds: float = 6.0,
        start_pose: np.ndarray | None = None,
        save_path: str | Path | None = None,
        num_samples: int = 1,
        num_frames: int | None = None,
        frozen_body: bool = False,
        spine3_aa: np.ndarray | None = None,
    ) -> np.ndarray:
        if frozen_body:
            raise NotImplementedError(
                "frozen_body is not supported with the kimodo backend."
            )
        self._align_fk_collar_to_pose(start_pose)
        n_frames = self._resolve_frames(motion_length_seconds, num_frames)
        body_pose, positions = self._run_worker(text, n_frames, num_samples, start_pose)

        if spine3_aa is not None:
            # Fixed-base path: route through positions so the trajectory lands
            # in the MPC's fixed spine3 frame (matches the MDM backend).
            arm_aa = self.smpl_positions_to_left_arm_trajectory(
                positions, spine3_aa=spine3_aa
            )
            return arm_aa[0] if num_samples == 1 else arm_aa

        arm_aa = smpl_body_pose_to_arm_aa(body_pose)  # (num_samples, n_frames, 3, 3)
        return arm_aa[0] if num_samples == 1 else arm_aa

    def generate_left_arm_position_samples(
        self,
        text: str,
        motion_length_seconds: float = 6.0,
        start_pose: np.ndarray | None = None,
        num_samples: int = 1,
        num_frames: int | None = None,
        frozen_body: bool = False,
    ) -> np.ndarray:
        if frozen_body:
            raise NotImplementedError(
                "frozen_body is not supported with the kimodo backend."
            )
        self._align_fk_collar_to_pose(start_pose)
        n_frames = self._resolve_frames(motion_length_seconds, num_frames)
        _body_pose, positions = self._run_worker(
            text, n_frames, num_samples, start_pose
        )
        return positions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_frames(motion_length_seconds: float, num_frames: int | None) -> int:
        return (
            int(num_frames)
            if num_frames is not None
            else int(motion_length_seconds * _FPS)
        )

    def _body_pose_to_visualizer_positions(self, body_pose: np.ndarray) -> np.ndarray:
        body_pose = np.asarray(body_pose, dtype=np.float64).reshape(21, 3)
        return smpl_body_pose_to_positions(body_pose, self._fk.tpose_all_joints)

    def _body_pose_batch_to_positions(self, body_pose: np.ndarray) -> np.ndarray:
        """FK ``(N, T, 21, 3)`` body_pose → ``(N, T, 22, 3)`` SMPL positions."""
        body_pose = np.asarray(body_pose, dtype=np.float64)
        n_samples, n_frames = body_pose.shape[0], body_pose.shape[1]
        out = np.empty((n_samples, n_frames, 22, 3), dtype=np.float64)
        for s in range(n_samples):
            for t in range(n_frames):
                out[s, t] = smpl_body_pose_to_positions(
                    body_pose[s, t], self._fk.tpose_all_joints
                )
        return out

    def _env_python(self) -> str:
        """Absolute path to the kimodo conda env's python interpreter.

        Resolved via ``conda env list --json`` (cached). Invoking the env's
        python directly avoids ``conda run -n``, which silently drops ``-n`` and
        runs in ``base`` when ``CONDA_PREFIX``/``CONDA_DEFAULT_ENV`` are set
        (e.g. under ``uv run``).
        """
        if self._python is not None:
            return self._python
        out = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
            env=_clean_env(),
        )
        for prefix in json.loads(out.stdout)["envs"]:
            if Path(prefix).name == self._conda_env:
                self._python = str(Path(prefix) / "bin" / "python")
                return self._python
        raise RuntimeError(
            f"conda env '{self._conda_env}' not found (conda env list). "
            "Create it per README 'Kimodo backend setup'."
        )

    def _run_worker(
        self,
        text: str,
        n_frames: int,
        num_samples: int,
        start_pose: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run kimodo in the isolated env; return body_pose and positions."""
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "kimodo_out.npz"
            cmd = [
                self._env_python(),
                str(self._worker_path),
                "--text",
                text,
                "--num_frames",
                str(n_frames),
                "--num_samples",
                str(num_samples),
                "--model_name",
                self._model_name,
                "--output_path",
                str(out_path),
                "--seed",
                str(self._seed),
                "--num_denoising_steps",
                str(self._num_denoising_steps),
            ]
            if start_pose is not None:
                start_positions = self._body_pose_to_visualizer_positions(start_pose)
                start_positions_path = Path(tmp) / "start_positions.npy"
                np.save(start_positions_path, start_positions)
                cmd += ["--start_positions_path", str(start_positions_path)]

            # Inherit stdout/stderr so the worker streams live — in particular
            # kimodo's tqdm denoising bar (it writes to stderr), which
            # capture_output would buffer and hide until/unless the run fails.
            result = subprocess.run(cmd, check=False, env=_clean_env())
            if result.returncode != 0:
                raise RuntimeError(
                    f"kimodo worker failed (exit {result.returncode}); "
                    "see the worker output above."
                )
            data = np.load(out_path)
            return (
                np.asarray(data["body_pose"], dtype=np.float64),
                np.asarray(data["positions"], dtype=np.float64),
            )
