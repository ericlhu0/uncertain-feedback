"""Agent (``agent``) cost generator that delegates to the ``codex`` CLI.

Rather than calling the LLM directly, this writes the task (the same prompt the
other backends use) into the run directory and lets the external ``codex`` coding
agent author the answer. To keep one consistent contract across all three
backends, codex is asked to emit the *same JSON object* (``description`` / ``code``
/ ``params`` / ...) into ``response.json``; that file is then parsed and compiled
exactly like the ``llm`` and ``turns`` outputs.

When an :class:`EvalState` is available (a planner with a Cartesian goal, images
enabled), codex is also given a render script and the pickled state so it can
*itself* roll out the cost it just authored and overlay the result against the
user's correction — inspecting that image each turn and refining ``response.json``
before finalizing. The initial context overlay paths are listed in ``TASK.md``.
Codex is instructed to load those paths itself and to leave a public
``ITERATION_LOG.md`` explaining each visual comparison and revision.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from uncertain_feedback.planners.mpc.costs.cost_generator import (
    CostGenerator,
    evaluate_candidate_cost,
)
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostValidationError,
    GeneratedPythonCost,
)

_RESPONSE_FILE = "response.json"
_STATE_FILE = "state.pkl"
_COMPARISON_FILE = "comparison.png"
_ANGLES_FILE = "angles.png"
_ITERATION_LOG_FILE = "ITERATION_LOG.md"
_RENDER_SCRIPT = (
    Path(__file__).resolve().parents[3] / "experiments" / "render_cost_comparison.py"
)
_REPO_ROOT = _RENDER_SCRIPT.parents[3]

_CODEX_INSTRUCTION = (
    "Read TASK.md in this directory. Load every local image path listed there "
    f"before drafting the first answer, maintain {_ITERATION_LOG_FILE} exactly "
    f"as TASK.md requests, and write the final cost as the single JSON object "
    f"it specifies (keys: description, code, params, explanation, "
    f"recipient_explanation) into {_RESPONSE_FILE}."
)


class AgentCostGenerator(CostGenerator):
    """Cost generator that runs the ``codex`` CLI to author the cost JSON."""

    def __init__(
        self, *args: Any, codex_cmd: str = "codex exec", **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.codex_cmd = codex_cmd

    def generate(self, install: bool = False) -> GeneratedPythonCost | None:
        try:
            self.begin()
            prompt_text, image_input = self.build_prompt()
            iterate = self.use_images and self.eval_state is not None
            if iterate:
                self.eval_state.save(self.run_dir / _STATE_FILE)
            (self.run_dir / "TASK.md").write_text(
                self._task_md(
                    prompt_text,
                    iterate=iterate,
                    image_input=image_input if self.use_images else None,
                ),
                encoding="utf-8",
            )

            self._run_codex()
            self._record_iteration_log(required=self.use_images)

            response_path = self.run_dir / _RESPONSE_FILE
            if not response_path.exists():
                raise GeneratedCostValidationError(
                    f"codex did not produce {_RESPONSE_FILE}; see codex.log"
                )
            raw = response_path.read_text(encoding="utf-8")
            (self.run_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
            response, cost = self.parse_cost(raw)
            score, _ = evaluate_candidate_cost(self.context, cost, self.rollout_fn)
            with open(self.run_dir / "score.json", "w", encoding="utf-8") as f:
                json.dump({"score": score}, f, indent=2)
            print(f"[cost-gen][agent] score={score:.4f}")
            self.save_response(response)
            if install:
                self.install(cost)
            self._on_success(cost, installed=install)
            return cost
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._on_failure(exc)
            return None

    def _task_md(
        self,
        prompt_text: str,
        *,
        iterate: bool,
        image_input: list[str] | None,
    ) -> str:
        archive_args = (
            " --archive-dir candidates --save-video"
            if self.save_candidate_videos
            else ""
        )
        header = (
            "# Cost-generation task\n\n"
            "Follow the prompt below and write the resulting JSON object to "
            f"`{_RESPONSE_FILE}` in this directory. The JSON schema and the cost "
            "function contract are described in the prompt.\n\n"
        )
        if image_input:
            image_lines = "\n".join(f"- `{path}`" for path in image_input)
            header += (
                "## Visual context files\n\n"
                "Open these local image files when judging the generated motion:\n"
                f"{image_lines}\n\n"
            )
        if self.use_images:
            header += (
                "## Required visual reasoning log\n\n"
                f"Maintain `{_ITERATION_LOG_FILE}` while you work. Before writing "
                f"the first `{_RESPONSE_FILE}`, load every local image path listed "
                "above and write what you observed. After each rollout-comparison "
                f"render, load `{_COMPARISON_FILE}` and `{_ANGLES_FILE}` and append "
                "one entry with:\n"
                "- which image files you loaded;\n"
                "- what visual mismatch or limitation you saw;\n"
                "- the candidate score when available;\n"
                "- what you changed in the next cost;\n"
                "- why that change should improve the rollout;\n"
                "- when you stop, whether you stopped because the generated arm "
                "movement matches the reference well enough or because you "
                "determined it cannot be made to match with the available cost "
                "API.\n\n"
            )
        if iterate:
            header += (
                "## Iterate using the rollout comparison\n\n"
                "After writing `response.json`, check how the cost you authored "
                "actually steers the arm by running, from this directory:\n\n"
                "```\n"
                f"uv run --project {_REPO_ROOT} python {_RENDER_SCRIPT} "
                f"--state {_STATE_FILE} --response {_RESPONSE_FILE} "
                f"--out {_COMPARISON_FILE} --angles-out {_ANGLES_FILE}"
                f"{archive_args}\n"
                "```\n\n"
                f"This prints an L2 score (lower is better) and writes "
                f"the comparison images. When the command includes an archive "
                f"directory, it also saves candidate JSON, score, rollout, "
                f"and MP4 files there. It writes "
                f"`{_COMPARISON_FILE}`, which overlays the motion your cost produced "
                "(red, 'cost rollout') against the entire intended corrected path "
                "(green, 'target corrected path': the pre-correction motion, the "
                "correction, and the continuation to the goal) it should match, and "
                f"`{_ANGLES_FILE}`, which plots each arm joint angle over time for "
                "the same two trajectories (green target vs red rollout) so you can "
                "compare the shape of the movement, not just endpoints. Open and "
                "look at both images: where the red arm/curves diverge from the "
                "green ones tells you what to fix. Revise "
                "`response.json`, re-run the command, and keep iterating until "
                "you are satisfied that the generated arm movement matches the "
                "green reference well enough, or until you determine there is no "
                "way to make it match with the available cost API. In your final "
                f"`{_ITERATION_LOG_FILE}` entry, state which stopping condition "
                "you reached and why. Leave the best `response.json` as your final "
                "answer.\n\n"
            )
        return f"{header}---\n\n{prompt_text}\n"

    def _run_codex(self) -> None:
        cmd = shlex.split(self.codex_cmd) + [_CODEX_INSTRUCTION]
        try:
            result = subprocess.run(
                cmd,
                cwd=self.run_dir,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GeneratedCostValidationError(
                f"codex CLI not found (command: {self.codex_cmd!r}). Install codex "
                "or set llm_cost.codex_cmd to its path."
            ) from exc
        log_text = "$ " + shlex.join(cmd) + "\n\n"
        log_text += (result.stdout or "") + (result.stderr or "")
        (self.run_dir / "codex.log").write_text(log_text, encoding="utf-8")
        if result.returncode != 0:
            raise GeneratedCostValidationError(
                f"codex exited with code {result.returncode}; see codex.log"
            )

    def _record_iteration_log(self, *, required: bool) -> None:
        iteration_log_path = self.run_dir / _ITERATION_LOG_FILE
        codex_log_path = self.run_dir / "codex.log"
        if iteration_log_path.exists():
            iteration_log = iteration_log_path.read_text(encoding="utf-8")
            with open(codex_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n===== {_ITERATION_LOG_FILE} =====\n")
                f.write(iteration_log)
                if not iteration_log.endswith("\n"):
                    f.write("\n")
            return
        if required:
            with open(codex_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n[missing] {_ITERATION_LOG_FILE}\n")
            raise GeneratedCostValidationError(
                f"codex did not produce {_ITERATION_LOG_FILE}; see codex.log"
            )
