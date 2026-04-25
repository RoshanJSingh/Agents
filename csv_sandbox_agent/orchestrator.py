from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .agents import DebuggerAgent, EngineerAgent, ProfilerAgent
from .code_extraction import CodeExtractionError, extract_python_code
from .code_validation import CodeValidationError, validate_generated_code
from .config import RetryConfig, SandboxConfig, create_run_dir
from .docker_runner import DockerRunner
from .error_compaction import compact_error, failure_fingerprint
from .llm_client import LLMClient
from .schemas import (
    DockerRunResult,
    DockerSandboxError,
    LLMResponseError,
    OrchestrationResult,
    ProfilerError,
    ProfilerReport,
    model_to_dict,
)


@dataclass
class RetryCircuitBreaker:
    max_retries: int = 5
    repeated_fingerprint_limit: int = 3
    invalid_python_limit: int = 2
    timeout_limit: int = 2
    missing_output_limit: int = 2
    failed_code_hashes: set[str] = field(default_factory=set)
    fingerprint_counts: dict[str, int] = field(default_factory=dict)
    invalid_python_streak: int = 0
    timeout_count: int = 0
    missing_output_count: int = 0

    @classmethod
    def from_config(cls, config: RetryConfig) -> "RetryCircuitBreaker":
        return cls(
            max_retries=config.max_retries,
            repeated_fingerprint_limit=config.repeated_fingerprint_limit,
            invalid_python_limit=config.invalid_python_limit,
            timeout_limit=config.timeout_limit,
            missing_output_limit=config.missing_output_limit,
        )

    def should_stop_for_code_hash(self, code_hash: str) -> str | None:
        if code_hash and code_hash in self.failed_code_hashes:
            return "generated code hash is identical to a previous failed attempt"
        return None

    def record_failure(
        self,
        *,
        attempt_index: int,
        code_hash: str,
        fingerprint: str,
        invalid_python: bool = False,
        timed_out: bool = False,
        missing_outputs: bool = False,
        compacted_error: str = "",
    ) -> str | None:
        if code_hash:
            self.failed_code_hashes.add(code_hash)

        self.fingerprint_counts[fingerprint] = self.fingerprint_counts.get(fingerprint, 0) + 1
        self.invalid_python_streak = (
            self.invalid_python_streak + 1 if invalid_python else 0
        )
        if timed_out:
            self.timeout_count += 1
        if missing_outputs:
            self.missing_output_count += 1

        if not compacted_error.strip():
            return "compacted error is empty; no actionable failure signal is available"
        if attempt_index >= self.max_retries:
            return "max retries reached"
        if self.fingerprint_counts[fingerprint] >= self.repeated_fingerprint_limit:
            return "same normalized failure fingerprint repeated too many times"
        if self.invalid_python_streak >= self.invalid_python_limit:
            return "debugger returned invalid Python twice in a row"
        if self.timeout_count >= self.timeout_limit:
            return "Docker execution timed out twice"
        if self.missing_output_count >= self.missing_output_limit:
            return "generated script exited 0 but missed required outputs twice"
        return None


def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


class CSVSandboxOrchestrator:
    def __init__(
        self,
        llm_client: LLMClient,
        docker_runner: DockerRunner | None = None,
        retry_config: RetryConfig | None = None,
        sandbox_config: SandboxConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.retry_config = retry_config or RetryConfig()
        self.docker_runner = docker_runner or DockerRunner(sandbox_config)
        self.profiler_agent = ProfilerAgent(llm_client)
        self.engineer_agent = EngineerAgent(llm_client)
        self.debugger_agent = DebuggerAgent(llm_client)

    def profile(self, input_csv: Path, workspace_root: Path) -> Path:
        run_dir = self._prepare_run_dir(input_csv, workspace_root)
        self.docker_runner.build_image_if_needed()
        result = self.docker_runner.run_profiler(run_dir)
        _write_text(run_dir / "profiler.stdout.log", result.stdout)
        _write_text(run_dir / "profiler.stderr.log", result.stderr)
        _write_json(run_dir / "profiler_execution.json", _model_dump(result))
        raw_profile_path = run_dir / "raw_profile.json"
        if result.exit_code != 0 or not raw_profile_path.exists():
            raise ProfilerError(
                f"Docker profiler failed with exit code {result.exit_code}. "
                f"Logs: {run_dir / 'profiler.stderr.log'}"
            )
        _read_json(raw_profile_path)
        return run_dir

    def run(
        self,
        input_csv: Path,
        workspace_root: Path,
        *,
        target: str | None = None,
        max_retries: int | None = None,
        timeout_seconds: int = 180,
    ) -> OrchestrationResult:
        if max_retries is not None:
            retry_config = RetryConfig(max_retries=max_retries)
        else:
            retry_config = self.retry_config
        breaker = RetryCircuitBreaker.from_config(retry_config)

        run_dir = self._prepare_run_dir(input_csv, workspace_root)
        history: list[dict[str, Any]] = []
        last_fingerprint: str | None = None
        last_compacted_error: str | None = None

        try:
            self.docker_runner.build_image_if_needed()
            profile_result = self.docker_runner.run_profiler(run_dir)
            _write_text(run_dir / "profiler.stdout.log", profile_result.stdout)
            _write_text(run_dir / "profiler.stderr.log", profile_result.stderr)
            _write_json(run_dir / "profiler_execution.json", _model_dump(profile_result))
            if profile_result.exit_code != 0 or not (run_dir / "raw_profile.json").exists():
                compacted = compact_error(profile_result.stdout, profile_result.stderr)
                return self._failure_result(
                    run_dir=run_dir,
                    attempts_used=0,
                    message="Docker profiler failed",
                    fingerprint=failure_fingerprint(
                        compacted_error=compacted,
                        exit_code=profile_result.exit_code,
                        timed_out=profile_result.timed_out,
                    ),
                    compacted_error=compacted,
                )

            raw_profile = _read_json(run_dir / "raw_profile.json")
            raw_profile_for_llm = dict(raw_profile)
            raw_profile_for_llm["requested_target"] = target

            profiler_report = self.profiler_agent.analyze(raw_profile_for_llm, run_dir)
            _write_json(run_dir / "profiler_report.json", model_to_dict(profiler_report))

            raw_response = self.engineer_agent.generate(
                raw_profile_for_llm,
                profiler_report,
                target,
                run_dir,
            )
        except (DockerSandboxError, LLMResponseError, ProfilerError, OSError, ValueError) as exc:
            compacted = str(exc)
            fp = failure_fingerprint(compacted_error=compacted, exit_code=-1)
            return self._failure_result(
                run_dir=run_dir,
                attempts_used=0,
                message=f"Pipeline setup failed: {exc}",
                fingerprint=fp,
                compacted_error=compacted,
            )

        attempt = 0
        while attempt <= retry_config.max_retries:
            _write_text(run_dir / f"raw_llm_response_attempt_{attempt}.txt", raw_response)
            code = ""
            code_hash = hash_code(raw_response)
            invalid_python = False
            missing_outputs = False
            result: DockerRunResult | None = None

            try:
                code = extract_python_code(raw_response)
                code_hash = hash_code(code)
                stop_reason = breaker.should_stop_for_code_hash(code_hash)
                if stop_reason:
                    compacted = stop_reason
                    _write_text(run_dir / f"compacted_error_attempt_{attempt}.txt", compacted)
                    fp = failure_fingerprint(compacted_error=compacted, exit_code=-1)
                    last_fingerprint = fp
                    last_compacted_error = compacted
                    record = self._history_record(
                        attempt=attempt,
                        code_hash=code_hash,
                        exit_code=-1,
                        compacted_error=compacted,
                        fingerprint=fp,
                        stop_reason=stop_reason,
                    )
                    history.append(record)
                    _append_jsonl(run_dir / "debug_history.jsonl", record)
                    _write_failure_fingerprints(run_dir, breaker, last_fingerprint)
                    return self._failure_result(
                        run_dir=run_dir,
                        attempts_used=attempt + 1,
                        message=stop_reason,
                        fingerprint=fp,
                        compacted_error=compacted,
                    )

                _write_text(run_dir / f"generated_script_attempt_{attempt}.py", code)
                validate_generated_code(code)
                _write_json(
                    run_dir / f"validation_attempt_{attempt}.json",
                    {"valid": True, "code_hash": code_hash},
                )
                _write_text(run_dir / "generated_script.py", code)

                result = self.docker_runner.run_generated_script(
                    run_dir,
                    target=target,
                    timeout_seconds=timeout_seconds,
                )
                _write_text(
                    run_dir / f"execution_attempt_{attempt}.stdout.log",
                    result.stdout,
                )
                _write_text(
                    run_dir / f"execution_attempt_{attempt}.stderr.log",
                    result.stderr,
                )
                _write_json(
                    run_dir / f"execution_attempt_{attempt}.json",
                    _model_dump(result),
                )

                if result.exit_code == 0:
                    output_check = self._validate_outputs(run_dir)
                    if output_check["ok"]:
                        outputs = output_check["outputs"]
                        final = OrchestrationResult(
                            status="success",
                            run_dir=run_dir,
                            attempts_used=attempt + 1,
                            message="Pipeline completed successfully",
                            outputs=outputs,
                        )
                        _write_json(run_dir / "orchestrator_result.json", _model_dump(final))
                        _write_failure_fingerprints(run_dir, breaker, None)
                        return final
                    missing_outputs = True
                    compacted = output_check["message"]
                else:
                    compacted = compact_error(result.stdout, result.stderr)

                fp = failure_fingerprint(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    compacted_error=compacted,
                    exit_code=result.exit_code,
                    timed_out=result.timed_out,
                )
                last_fingerprint = fp
                last_compacted_error = compacted

            except CodeExtractionError as exc:
                invalid_python = True
                compacted = f"Code extraction failed: {exc}"
                fp = failure_fingerprint(compacted_error=compacted, exit_code=-1)
                _write_json(
                    run_dir / f"validation_attempt_{attempt}.json",
                    {"valid": False, "stage": "extraction", "error": str(exc)},
                )
                last_fingerprint = fp
                last_compacted_error = compacted
            except CodeValidationError as exc:
                compacted = f"Code validation failed: {exc}"
                fp = failure_fingerprint(compacted_error=compacted, exit_code=-1)
                _write_json(
                    run_dir / f"validation_attempt_{attempt}.json",
                    {
                        "valid": False,
                        "stage": "validation",
                        "error": str(exc),
                        "code_hash": code_hash,
                    },
                )
                last_fingerprint = fp
                last_compacted_error = compacted
            except DockerSandboxError as exc:
                compacted = f"Docker sandbox execution failed: {exc}"
                fp = failure_fingerprint(compacted_error=compacted, exit_code=125)
                last_fingerprint = fp
                last_compacted_error = compacted

            stop_reason = breaker.record_failure(
                attempt_index=attempt,
                code_hash=code_hash,
                fingerprint=last_fingerprint or "",
                invalid_python=invalid_python,
                timed_out=bool(result.timed_out if result else False),
                missing_outputs=missing_outputs,
                compacted_error=last_compacted_error or "",
            )
            _write_text(
                run_dir / f"compacted_error_attempt_{attempt}.txt",
                last_compacted_error or "",
            )
            record = self._history_record(
                attempt=attempt,
                code_hash=code_hash,
                exit_code=result.exit_code if result else -1,
                compacted_error=last_compacted_error or "",
                fingerprint=last_fingerprint or "",
                timed_out=bool(result.timed_out if result else False),
                debugger_summary=None,
                stop_reason=stop_reason,
            )
            history.append(record)
            _write_failure_fingerprints(run_dir, breaker, last_fingerprint)

            if stop_reason:
                _append_jsonl(run_dir / "debug_history.jsonl", record)
                return self._failure_result(
                    run_dir=run_dir,
                    attempts_used=attempt + 1,
                    message=stop_reason,
                    fingerprint=last_fingerprint,
                    compacted_error=last_compacted_error,
                )

            try:
                raw_response = self.debugger_agent.debug(
                    code=code or raw_response,
                    compact_error=last_compacted_error or "",
                    profiler_report=profiler_report,
                    history=history,
                    target=target,
                    audit_dir=run_dir,
                    attempt=attempt,
                )
                record["debugger_summary"] = (
                    f"debugger response saved to llm_debugger_response_attempt_{attempt}.txt"
                )
                _append_jsonl(run_dir / "debug_history.jsonl", record)
            except LLMResponseError as exc:
                record["debugger_summary"] = f"debugger failed: {exc}"
                record["stop_reason"] = "debugger LLM call failed"
                _append_jsonl(run_dir / "debug_history.jsonl", record)
                return self._failure_result(
                    run_dir=run_dir,
                    attempts_used=attempt + 1,
                    message=f"Debugger LLM call failed: {exc}",
                    fingerprint=last_fingerprint,
                    compacted_error=last_compacted_error,
                )

            attempt += 1

        return self._failure_result(
            run_dir=run_dir,
            attempts_used=attempt,
            message="max retries reached",
            fingerprint=last_fingerprint,
            compacted_error=last_compacted_error,
        )

    def _prepare_run_dir(self, input_csv: Path, workspace_root: Path) -> Path:
        source = input_csv.expanduser().resolve()
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Input CSV not found: {source}")
        run_dir = create_run_dir(workspace_root)
        shutil.copy2(source, run_dir / "input.csv")
        return run_dir

    def _validate_outputs(self, run_dir: Path) -> dict[str, Any]:
        output_dir = run_dir / "output"
        required = ["cleaned_data.csv", "metrics.json", "manifest.json"]
        missing = [name for name in required if not (output_dir / name).exists()]
        if missing:
            return {
                "ok": False,
                "message": "Generated script exited 0 but missing required outputs: "
                + ", ".join(missing),
                "outputs": {},
            }
        try:
            metrics = _read_json(output_dir / "metrics.json")
            manifest = _read_json(output_dir / "manifest.json")
        except Exception as exc:
            return {
                "ok": False,
                "message": f"Generated script wrote invalid JSON outputs: {exc}",
                "outputs": {},
            }

        model_trained = bool(
            manifest.get("model_trained", False) or metrics.get("model_trained", False)
        )
        if model_trained and not (output_dir / "model.pkl").exists():
            return {
                "ok": False,
                "message": "Generated script reported model_trained true but model.pkl is missing",
                "outputs": {},
            }

        outputs = {
            name: str((output_dir / name).resolve())
            for name in ["cleaned_data.csv", "metrics.json", "manifest.json", "model.pkl"]
            if (output_dir / name).exists()
        }
        return {"ok": True, "message": "outputs verified", "outputs": outputs}

    def _history_record(
        self,
        *,
        attempt: int,
        code_hash: str,
        exit_code: int,
        compacted_error: str,
        fingerprint: str,
        timed_out: bool = False,
        debugger_summary: str | None = None,
        stop_reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "attempt": attempt,
            "code_hash": code_hash,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "compacted_error": compacted_error,
            "fingerprint": fingerprint,
            "debugger_summary": debugger_summary,
            "stop_reason": stop_reason,
        }

    def _failure_result(
        self,
        *,
        run_dir: Path,
        attempts_used: int,
        message: str,
        fingerprint: str | None,
        compacted_error: str | None,
    ) -> OrchestrationResult:
        result = OrchestrationResult(
            status="failed",
            run_dir=run_dir,
            attempts_used=attempts_used,
            last_failure_fingerprint=fingerprint,
            last_compacted_error=compacted_error,
            message=message,
        )
        _write_json(run_dir / "orchestrator_result.json", _model_dump(result))
        return result


def _model_dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[attr-defined]
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _write_text(path: Path, text: str) -> None:
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, default=str) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_failure_fingerprints(
    run_dir: Path,
    breaker: RetryCircuitBreaker,
    last_fingerprint: str | None,
) -> None:
    _write_json(
        run_dir / "failure_fingerprints.json",
        {
            "last_failure_fingerprint": last_fingerprint,
            "fingerprint_counts": breaker.fingerprint_counts,
            "failed_code_hashes": sorted(breaker.failed_code_hashes),
        },
    )
