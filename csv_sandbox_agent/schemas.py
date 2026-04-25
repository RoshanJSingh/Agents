from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

try:  # Pydantic v2
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - Pydantic v1 fallback
    ConfigDict = None  # type: ignore[assignment]


class DockerSandboxError(RuntimeError):
    """Raised when Docker sandbox setup or execution fails."""


class LLMResponseError(RuntimeError):
    """Raised when an LLM response cannot be parsed or validated."""


class OrchestrationError(RuntimeError):
    """Raised when the full pipeline cannot complete."""


class ProfilerError(RuntimeError):
    """Raised when Docker-based profiling fails or emits invalid metadata."""


class ColumnIssue(BaseModel):
    column: str
    issue_type: Literal[
        "missing_values",
        "bad_date_format",
        "currency_or_numeric_text",
        "categorical_encoding_needed",
        "high_cardinality",
        "mixed_type",
        "duplicate_column",
        "target_problem",
        "constant_or_near_constant",
        "other",
    ]
    severity: Literal["low", "medium", "high"]
    evidence: str
    recommended_action: str

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - Pydantic v1 fallback
        class Config:
            extra = "forbid"


class ProfilerReport(BaseModel):
    dataset_summary: str
    likely_task_type: Literal["classification", "regression", "unknown"]
    target_column: str | None = None
    column_issues: list[ColumnIssue] = Field(default_factory=list)
    global_recommendations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - Pydantic v1 fallback
        class Config:
            extra = "forbid"


class DockerRunResult(BaseModel):
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_seconds: float = 0.0

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - Pydantic v1 fallback
        class Config:
            extra = "forbid"


class OrchestrationResult(BaseModel):
    status: Literal["success", "failed"]
    run_dir: Path
    attempts_used: int
    last_failure_fingerprint: str | None = None
    last_compacted_error: str | None = None
    message: str
    outputs: dict[str, str] = Field(default_factory=dict)

    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)
    else:  # pragma: no cover - Pydantic v1 fallback
        class Config:
            arbitrary_types_allowed = True


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    """Return a dict for either Pydantic v1 or v2."""

    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def validate_model(model_cls: type[BaseModel], data: Any) -> BaseModel:
    """Validate data with either Pydantic v1 or v2."""

    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)  # type: ignore[attr-defined]
    return model_cls.parse_obj(data)
