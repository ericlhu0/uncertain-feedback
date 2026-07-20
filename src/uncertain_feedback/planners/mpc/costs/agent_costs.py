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
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Thread
from typing import Any, TextIO

from uncertain_feedback.planners.mpc.costs.cost_generator import (
    CostGenerator,
    evaluate_candidate_cost,
    rank_candidate_cost,
)
from uncertain_feedback.planners.mpc.costs.generated import (
    GeneratedCostValidationError,
    GeneratedPythonCost,
)
from uncertain_feedback.planners.mpc.costs.prompts import (
    build_staged_task_body,
    corpus_grounding_note,
    corpus_task_section,
)

_RESPONSE_FILE = "response.json"
_STATE_FILE = "state.pkl"
_COMPARISON_FILE = "comparison.png"
_ANGLES_FILE = "angles.png"
_ITERATION_LOG_FILE = "ITERATION_LOG.md"
_STAGE_LOG_FILE = "stage_log.md"
_CODEX_POLL_INTERVAL_SECONDS = 30.0
_CODEX_TIMEOUT_SECONDS = 30.0 * 60.0
_CODEX_TERMINATE_GRACE_SECONDS = 5.0
_RENDER_SCRIPT = (
    Path(__file__).resolve().parents[3] / "experiments" / "render_cost_comparison.py"
)
_REPO_ROOT = _RENDER_SCRIPT.parents[3]
_SANDBOX_WORKSPACE = Path("/tmp/workspace")
_SANDBOX_RUNTIME = Path("/tmp/runtime")
_SANDBOX_VENV = Path("/tmp/venv")
_SANDBOX_CODEX_HOME = Path("/tmp/codex-home")

_CODEX_INSTRUCTION = (
    "Read TASK.md in this directory. Load every local image path listed there "
    f"before drafting the first answer, maintain {_ITERATION_LOG_FILE} exactly "
    f"as TASK.md requests, maintain {_STAGE_LOG_FILE} exactly as TASK.md "
    f"requests, and write the final cost as the single JSON object it specifies "
    f"(keys: description, code, params, explanation, recipient_explanation) into "
    f"{_RESPONSE_FILE}."
)


def _stage_section(stage_log: str, heading: str) -> str | None:
    """Return one stage-log section, ending before the next level-two heading."""
    start = stage_log.find(heading)
    if start < 0:
        return None
    start += len(heading)
    end = stage_log.find("\n## ", start)
    return (stage_log[start:] if end < 0 else stage_log[start:end]).strip()


class AgentCostGenerator(CostGenerator):
    """Cost generator that runs the ``codex`` CLI to author the cost JSON."""

    _codex_log_label = "agent"
    _stream_codex_output = False

    def __init__(
        self,
        *args: Any,
        codex_cmd: str = "codex exec",
        timeout_seconds: float | None = None,
        corpus_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, corpus_dir=corpus_dir, **kwargs)
        self.codex_cmd = codex_cmd
        self.timeout_seconds = timeout_seconds
        self._staged_corpus_dir: Path | None = None

    def _resolved_corpus_dir(self) -> Path | None:
        if self.corpus_dir is None:
            return None
        resolved = self.corpus_dir.resolve()
        return resolved if (resolved / "manifest.json").exists() else None

    def _stage_corpus(self) -> Path | None:
        if self._staged_corpus_dir is not None:
            return self._staged_corpus_dir
        corpus_dir = self._resolved_corpus_dir()
        if corpus_dir is None:
            return None
        destination = self.run_dir / "inputs" / "corpus"
        destination.mkdir(parents=True, exist_ok=True)
        entries = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
        staged_entries = []
        for entry in entries:
            staged = {
                key: value
                for key, value in entry.items()
                if key not in {"feedback_text", "trigger_violation"}
            }
            for field in ("traj_file", "features_file"):
                source = (corpus_dir / entry[field]).resolve()
                if not source.is_relative_to(corpus_dir):
                    raise GeneratedCostValidationError(
                        f"corpus {field} leaves the corpus directory: {source}"
                    )
                filename = Path(entry[field]).name
                shutil.copy2(source, destination / filename)
                staged[field] = filename
            staged_entries.append(staged)
        (destination / "manifest.json").write_text(
            json.dumps(staged_entries, indent=2), encoding="utf-8"
        )
        self._staged_corpus_dir = _SANDBOX_WORKSPACE / "inputs" / "corpus"
        return self._staged_corpus_dir

    def _stage_files(self, paths: list[Path], directory: str) -> list[Path]:
        destination = self.run_dir / "inputs" / directory
        destination.mkdir(parents=True, exist_ok=True)
        staged_paths = []
        for index, source in enumerate(paths):
            filename = f"{index:02d}_{source.name}"
            target = destination / filename
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            staged_paths.append(_SANDBOX_WORKSPACE / "inputs" / directory / filename)
        return staged_paths

    def _corpus_section(self) -> str:
        corpus_dir = self._stage_corpus()
        return "" if corpus_dir is None else corpus_task_section(corpus_dir)

    def _corpus_note(self) -> str | None:
        corpus_dir = self._stage_corpus()
        return None if corpus_dir is None else corpus_grounding_note(corpus_dir)

    def generate(self, install: bool = False) -> GeneratedPythonCost | None:
        try:
            self.begin()
            prompt_text, image_input = build_staged_task_body(
                self.instruction,
                self.summaries,
                self.images if self.use_images else {},
                corpus_note=self._corpus_note(),
            )
            staged_images = self._stage_files(image_input, "context")
            eval_state = self.eval_state
            iterate = self.use_images and eval_state is not None
            if eval_state is not None and iterate:
                eval_state.save(self.run_dir / _STATE_FILE)
            (self.run_dir / "TASK.md").write_text(
                self._task_md(
                    prompt_text,
                    iterate=iterate,
                    image_input=(
                        [str(path) for path in staged_images]
                        if self.use_images
                        else None
                    ),
                ),
                encoding="utf-8",
            )

            self._run_codex()
            self._record_stage_log()
            self._record_iteration_log(required=self.use_images)

            response_path = self.run_dir / _RESPONSE_FILE
            if not response_path.exists():
                raise GeneratedCostValidationError(
                    f"codex did not produce {_RESPONSE_FILE}; see codex.log"
                )
            raw = response_path.read_text(encoding="utf-8")
            (self.run_dir / "raw_response.txt").write_text(raw, encoding="utf-8")
            response, cost = self.parse_cost(raw)
            ranking = rank_candidate_cost(self.context, cost)
            score, _ = evaluate_candidate_cost(self.context, cost, self.rollout_fn)
            with open(self.run_dir / "score.json", "w", encoding="utf-8") as f:
                json.dump({"score": score}, f, indent=2)
            print(f"[cost-gen][agent] score={score:.4f}")
            self.save_response(response)
            stage_log = (self.run_dir / _STAGE_LOG_FILE).read_text(encoding="utf-8")
            self.save_rationale(
                response,
                interpret_raw=_stage_section(stage_log, "## Stage 1 response"),
                ground_raw=_stage_section(stage_log, "## Stage 2 response"),
                ranking=ranking,
            )
            self.require_original_plan_improvement(ranking)
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
            "function contract are described in the prompt. Treat this task, its "
            "listed images, and its sanitized corpus as the complete evidence set; "
            "simulator definitions and prior runs are intentionally unavailable.\n\n"
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
        corpus_section = self._corpus_section()
        corpus_check_line = (
            "- `## Corpus check` (before `## Stage 2 response`) — the numpy/pandas "
            "check of your candidate thresholds against the comfortable frames in the "
            "executed-trajectory corpus, with the per-entry worst-case margin.\n"
            if corpus_section
            else ""
        )
        header += (
            "## Required stage log\n\n"
            f"Maintain `{_STAGE_LOG_FILE}` while you work. It must show the response "
            "you produced for each stage prompt with exactly these headings:\n"
            "- `## Stage 1 response` — the interpretation JSON you wrote after "
            "reading the Stage 1 prompt and visual context.\n"
            f"{corpus_check_line}"
            "- `## Stage 2 response` — the numeric grounding/specification JSON you "
            "wrote from the Stage 2 prompt.\n"
            "- `## Stage 3 response` — the final cost JSON you wrote to "
            f"`{_RESPONSE_FILE}`.\n\n"
        )
        if corpus_section:
            header += f"{corpus_section}\n\n"
        if iterate:
            header += (
                "## Iterate using the rollout comparison\n\n"
                "After writing `response.json`, check how the cost you authored "
                "actually steers the arm by running, from this directory:\n\n"
                "```\n"
                f"{_SANDBOX_VENV}/bin/python "
                f"{_SANDBOX_RUNTIME}/src/uncertain_feedback/experiments/"
                "render_cost_comparison.py "
                f"--state {_STATE_FILE} --response {_RESPONSE_FILE} "
                f"--out {_COMPARISON_FILE} --angles-out {_ANGLES_FILE}"
                f"{archive_args}\n"
                "```\n\n"
                f"This prints an L2 score (lower is better) and a `goal reached:` "
                f"line reporting whether the arm still reaches the goal. Unless the "
                f"correction implies the person is uncomfortable reaching the goal that "
                f"way, keep revising until it reports the goal is reached — a cost that "
                f"blocks the goal is not acceptable. It also writes "
                f"the comparison images. When the command includes an archive "
                f"directory, it also saves candidate JSON, score, rollout, "
                f"and MP4 files there. It writes "
                f"`{_COMPARISON_FILE}`, which overlays the motion your cost produced "
                "(red, 'cost rollout') against the entire intended corrected path "
                "(green, 'target corrected path': the pre-correction motion, the "
                "correction, and the continuation to the goal) it should match, and "
                f"`{_ANGLES_FILE}`, which plots each arm joint angle over time for "
                "the same two trajectories (green target vs red rollout), plus the "
                "initial uncorrected goal-seeking path (dashed steel-blue) so you "
                "can see how the correction changed the motion, and compare the "
                "shape of the movement, not just endpoints. Open and "
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

    @staticmethod
    def _prepare_isolated_runtime(destination: Path) -> None:
        source = _REPO_ROOT / "src" / "uncertain_feedback"
        package = destination / "src" / "uncertain_feedback"
        package.mkdir(parents=True)
        ignored = shutil.ignore_patterns("__pycache__", "*artifacts*")
        for directory in ("planners", "uncertainty", "utils"):
            shutil.copytree(
                source / directory,
                package / directory,
                ignore=ignored,
            )
        experiments = package / "experiments"
        experiments.mkdir()
        shutil.copy2(_RENDER_SCRIPT, experiments / _RENDER_SCRIPT.name)

    def _isolated_command(  # pylint: disable=too-many-locals
        self,
        command: list[str],
        temporary_dir: Path,
        workspace_dir: Path | None = None,
    ) -> list[str]:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise GeneratedCostValidationError(
                "the agent backend requires bubblewrap so hidden simulator state "
                "cannot be read"
            )
        executable = shutil.which(command[0])
        if executable is None:
            raise GeneratedCostValidationError(
                f"codex CLI not found (command: {self.codex_cmd!r}). Install codex "
                "or set llm_cost.codex_cmd to its path."
            )

        runtime = temporary_dir / "runtime"
        codex_home = temporary_dir / "codex-home"
        workspace = self.run_dir.resolve() if workspace_dir is None else workspace_dir
        self._prepare_isolated_runtime(runtime)
        codex_home.mkdir()

        sandbox_command = list(command)
        sandbox_command[0] = executable
        extra_mounts: list[str] = []
        sandbox_path = f"{_SANDBOX_VENV}/bin:/usr/local/bin:/usr/bin:/bin"
        executable_source = Path(executable)
        executable_path = executable_source.resolve()
        project_venv = (_REPO_ROOT / ".venv").resolve()
        if executable_source.is_relative_to(project_venv):
            sandbox_command[0] = str(
                _SANDBOX_VENV / executable_source.relative_to(project_venv)
            )
        if Path(command[0]).name == "codex":
            source_codex_home = Path(
                os.environ.get("CODEX_HOME", Path.home() / ".codex")
            )
            auth_file = source_codex_home / "auth.json"
            if not auth_file.exists():
                raise GeneratedCostValidationError(
                    f"codex authentication not found at {auth_file}"
                )
            shutil.copy2(auth_file, codex_home / "auth.json")
            prompt = sandbox_command.pop()
            for flag in ("--ephemeral", "--ignore-user-config", "--ignore-rules"):
                if flag not in sandbox_command:
                    sandbox_command.append(flag)
            sandbox_command.append(prompt)

            if executable_path.is_relative_to(Path.home()):
                install_root = Path(executable).parent.parent.resolve()
                sandbox_install = Path("/tmp/codex-install")
                extra_mounts = [
                    "--dir",
                    str(sandbox_install),
                    "--ro-bind",
                    str(install_root),
                    str(sandbox_install),
                ]
                sandbox_command[0] = str(
                    sandbox_install / "bin" / Path(executable).name
                )
                sandbox_path = f"{sandbox_install}/bin:{sandbox_path}"

        environment = {
            "PATH": sandbox_path,
            "HOME": "/home/agent",
            "CODEX_HOME": str(_SANDBOX_CODEX_HOME),
            "PYTHONPATH": f"{_SANDBOX_RUNTIME}/src",
            "MPLBACKEND": "Agg",
        }
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "NODE_EXTRA_CA_CERTS",
        ):
            if name in os.environ:
                environment[name] = os.environ[name]
        environment_args = [
            argument
            for name, value in environment.items()
            for argument in ("--setenv", name, value)
        ]
        return [
            bwrap,
            "--ro-bind",
            "/",
            "/",
            "--tmpfs",
            "/home",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/home/agent",
            "--dir",
            str(_SANDBOX_WORKSPACE),
            "--bind",
            str(workspace),
            str(_SANDBOX_WORKSPACE),
            "--dir",
            str(_SANDBOX_RUNTIME),
            "--ro-bind",
            str(runtime),
            str(_SANDBOX_RUNTIME),
            "--dir",
            str(_SANDBOX_VENV),
            "--ro-bind",
            str((_REPO_ROOT / ".venv").resolve()),
            str(_SANDBOX_VENV),
            "--dir",
            str(_SANDBOX_CODEX_HOME),
            "--bind",
            str(codex_home),
            str(_SANDBOX_CODEX_HOME),
            *extra_mounts,
            "--clearenv",
            *environment_args,
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--die-with-parent",
            "--chdir",
            str(_SANDBOX_WORKSPACE),
            *sandbox_command,
        ]

    def _run_codex(self) -> None:  # pylint: disable=too-many-locals
        log_path = self.run_dir / "codex.log"
        log_prefix = f"[cost-gen][{self._codex_log_label}]"
        start = time.perf_counter()
        returncode: int | None = None
        output_thread: Thread | None = None
        timeout_seconds = (
            _CODEX_TIMEOUT_SECONDS
            if self.timeout_seconds is None
            else self.timeout_seconds
        )
        print(f"{log_prefix} live log: {log_path}", flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix="cost-agent-") as tmp:
                temporary_dir = Path(tmp)
                workspace = temporary_dir / "workspace"
                shutil.copytree(self.run_dir, workspace)
                command = shlex.split(self.codex_cmd) + [_CODEX_INSTRUCTION]
                cmd = self._isolated_command(command, temporary_dir, workspace)
                cmd_line = shlex.join(cmd)
                print(f"{log_prefix} starting codex: {cmd_line}", flush=True)
                with open(log_path, "w", encoding="utf-8") as log:
                    log.write("$ " + cmd_line + "\n\n")
                    log.flush()
                    process = subprocess.Popen(  # pylint: disable=consider-using-with
                        cmd,
                        cwd=self.run_dir,
                        stdout=(subprocess.PIPE if self._stream_codex_output else log),
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    if self._stream_codex_output:
                        assert process.stdout is not None
                        output_thread = Thread(
                            target=self._tee_codex_output,
                            args=(process.stdout, log, log_prefix),
                            daemon=True,
                        )
                        output_thread.start()
                    while True:
                        elapsed = time.perf_counter() - start
                        remaining = timeout_seconds - elapsed
                        if remaining <= 0.0:
                            message = (
                                f"{log_prefix} codex exceeded its timeout; "
                                "terminating child"
                            )
                            self._terminate_codex(process, output_thread, log, message)
                            raise GeneratedCostValidationError(
                                f"codex timed out after {timeout_seconds:g} seconds; "
                                "see codex.log"
                            )
                        try:
                            returncode = process.wait(
                                timeout=min(_CODEX_POLL_INTERVAL_SECONDS, remaining)
                            )
                            break
                        except subprocess.TimeoutExpired:
                            print(
                                f"{log_prefix} codex still running "
                                f"({time.perf_counter() - start:.0f}s); "
                                f"log: {log_path}",
                                flush=True,
                            )
                    if output_thread is not None:
                        output_thread.join()
                    log.flush()
                shutil.copytree(workspace, self.run_dir, dirs_exist_ok=True)
        except FileNotFoundError as exc:
            raise GeneratedCostValidationError(
                f"codex CLI not found (command: {self.codex_cmd!r}). Install codex "
                "or set llm_cost.codex_cmd to its path."
            ) from exc
        print(
            f"{log_prefix} codex finished in "
            f"{time.perf_counter() - start:.1f}s (exit {returncode})",
            flush=True,
        )
        if returncode is None:
            raise GeneratedCostValidationError("codex did not report an exit code.")
        if returncode != 0:
            raise GeneratedCostValidationError(
                f"codex exited with code {returncode}; see codex.log"
            )

    @staticmethod
    def _terminate_codex(
        process: subprocess.Popen[str],
        output_thread: Thread | None,
        log: TextIO,
        message: str,
    ) -> None:
        print(message, flush=True)
        process.terminate()
        try:
            process.wait(timeout=_CODEX_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=_CODEX_TERMINATE_GRACE_SECONDS)
        if output_thread is not None:
            output_thread.join()
        log.write(f"\n\n{message}\n")
        log.flush()

    @staticmethod
    def _tee_codex_output(
        output: TextIO,
        log: TextIO,
        log_prefix: str,
    ) -> None:
        started = False
        with output:
            for line in output:
                if not started:
                    print(f"{log_prefix} codex output:", flush=True)
                    started = True
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
        if started:
            print(flush=True)

    def _record_stage_log(self) -> None:
        stage_log_path = self.run_dir / _STAGE_LOG_FILE
        codex_log_path = self.run_dir / "codex.log"
        if not stage_log_path.exists():
            with open(codex_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n[missing] {_STAGE_LOG_FILE}\n")
            raise GeneratedCostValidationError(
                f"codex did not produce {_STAGE_LOG_FILE}; see codex.log"
            )
        stage_log = stage_log_path.read_text(encoding="utf-8")
        with open(codex_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n===== {_STAGE_LOG_FILE} =====\n")
            f.write(stage_log)
            if not stage_log.endswith("\n"):
                f.write("\n")

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
