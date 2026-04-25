from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .llm_client import LLMClient
from .prompts import (
    build_debugger_prompt,
    build_engineer_prompt,
    build_profiler_prompt,
)
from .schemas import LLMResponseError, ProfilerReport, model_to_dict, validate_model


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class ProfilerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def analyze(self, raw_profile: dict[str, Any], audit_dir: Path) -> ProfilerReport:
        system, user = build_profiler_prompt(raw_profile)
        (audit_dir / "profiler_prompt_metadata.json").write_text(
            json.dumps(
                {
                    "system_chars": len(system),
                    "user_chars": len(user),
                    "profile_columns": len(raw_profile.get("columns", []))
                    if isinstance(raw_profile.get("columns"), list)
                    else None,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        raw = self.llm.complete(system=system, user=user, temperature=0.0)
        _write_text(audit_dir / "llm_profiler_response_0.json", raw)
        try:
            return self._parse(raw)
        except Exception as first_exc:
            repair_system = (
                "You repair invalid JSON. Return JSON only, no markdown, no prose. "
                "The JSON must satisfy the profiler report schema exactly."
            )
            repair_user = (
                f"Validation or parsing failed with:\n{first_exc}\n\n"
                f"Original raw profile:\n{json.dumps(raw_profile, default=str)}\n\n"
                f"Invalid response:\n{raw}"
            )
            (audit_dir / "profiler_repair_prompt_metadata.json").write_text(
                json.dumps(
                    {
                        "system_chars": len(repair_system),
                        "user_chars": len(repair_user),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            repaired = self.llm.complete(
                system=repair_system,
                user=repair_user,
                temperature=0.0,
            )
            _write_text(audit_dir / "llm_profiler_response_1_repair.json", repaired)
            try:
                return self._parse(repaired)
            except Exception as second_exc:
                error_payload = {
                    "error": str(second_exc),
                    "first_error": str(first_exc),
                    "raw_response": raw,
                    "repair_response": repaired,
                }
                _write_text(
                    audit_dir / "profiler_report_error.json",
                    json.dumps(error_payload, indent=2, default=str),
                )
                raise LLMResponseError(
                    f"Profiler agent returned invalid JSON twice: {second_exc}"
                ) from second_exc

    @staticmethod
    def _parse(raw: str) -> ProfilerReport:
        data = json.loads(raw)
        report = validate_model(ProfilerReport, data)
        if not isinstance(report, ProfilerReport):
            report = ProfilerReport.parse_obj(model_to_dict(report))
        return report


class EngineerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(
        self,
        raw_profile: dict[str, Any],
        profiler_report: ProfilerReport,
        target: str | None,
        audit_dir: Path,
    ) -> str:
        system, user = build_engineer_prompt(raw_profile, profiler_report, target)
        (audit_dir / "engineer_prompt_metadata.json").write_text(
            json.dumps(
                {
                    "system_chars": len(system),
                    "user_chars": len(user),
                    "target": target,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raw = self.llm.complete(system=system, user=user, temperature=0.0)
        _write_text(audit_dir / "llm_engineer_response_0.txt", raw)
        return raw


class DebuggerAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def debug(
        self,
        code: str,
        compact_error: str,
        profiler_report: ProfilerReport,
        history: list[dict[str, Any]],
        target: str | None,
        audit_dir: Path,
        attempt: int,
    ) -> str:
        system, user = build_debugger_prompt(
            code=code,
            compact_error=compact_error,
            profiler_report=profiler_report,
            history=history,
            target=target,
        )
        (audit_dir / f"debugger_input_summary_attempt_{attempt}.json").write_text(
            json.dumps(
                {
                    "system_chars": len(system),
                    "user_chars": len(user),
                    "compact_error": compact_error,
                    "history_count": len(history),
                    "target": target,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        raw = self.llm.complete(system=system, user=user, temperature=0.0)
        _write_text(audit_dir / f"llm_debugger_response_attempt_{attempt}.txt", raw)
        return raw
