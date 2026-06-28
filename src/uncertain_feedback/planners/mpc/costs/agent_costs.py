"""Agent (``agent``) cost generator that delegates to the ``codex`` CLI.

Rather than calling the LLM directly, this writes the task (the same prompt the
other backends use) into the run directory and lets the external ``codex`` coding
agent author the answer. To keep one consistent contract across all three
backends, codex is asked to emit the *same JSON object* (``description`` / ``code``
/ ``params`` / ...) into ``response.json``; that file is then parsed and compiled
exactly like the ``llm`` and ``turns`` outputs.
"""

from __future__ import annotations

import json
import shlex
import subprocess
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

_CODEX_INSTRUCTION = (
    f"Read TASK.md in this directory and write your answer as the single JSON "
    f"object it specifies (keys: description, code, params, explanation, "
    f"recipient_explanation) into {_RESPONSE_FILE}. Write nothing else."
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
            prompt_text, _ = self.build_prompt()
            (self.run_dir / "TASK.md").write_text(
                self._task_md(prompt_text), encoding="utf-8"
            )

            self._run_codex()

            response_path = self.run_dir / _RESPONSE_FILE
            if not response_path.exists():
                raise GeneratedCostValidationError(
                    f"codex did not produce {_RESPONSE_FILE}; see codex.log"
                )
            raw = response_path.read_text(encoding="utf-8")
            (self.run_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
            response, cost = self.parse_cost(raw)
            score = evaluate_candidate_cost(self.context, cost, self.rollout_fn)
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

    def _task_md(self, prompt_text: str) -> str:
        return (
            "# Cost-generation task\n\n"
            "Follow the prompt below and write the resulting JSON object to "
            f"`{_RESPONSE_FILE}` in this directory. The JSON schema and the cost "
            "function contract are described in the prompt.\n\n"
            "---\n\n"
            f"{prompt_text}\n"
        )

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
        (self.run_dir / "codex.log").write_text(
            (result.stdout or "") + (result.stderr or ""), encoding="utf-8"
        )
        if result.returncode != 0:
            raise GeneratedCostValidationError(
                f"codex exited with code {result.returncode}; see codex.log"
            )
