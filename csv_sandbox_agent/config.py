from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


@dataclass(frozen=True)
class SandboxConfig:
    image_name: str = "csv-sandbox-agent:latest"
    dockerfile_path: Path = PROJECT_ROOT / "sandbox" / "Dockerfile"
    memory_limit: str = "1g"
    nano_cpus: int = 2_000_000_000
    pids_limit: int = 128
    profiler_timeout_seconds: int = 60
    execution_timeout_seconds: int = 180


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 5
    repeated_fingerprint_limit: int = 3
    invalid_python_limit: int = 2
    timeout_limit: int = 2
    missing_output_limit: int = 2


def create_run_dir(workspace_root: Path) -> Path:
    workspace_root = workspace_root.expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y_%m_%d_%H%M%S")
    suffix = uuid4().hex[:6]
    run_dir = workspace_root / f"{stamp}_{suffix}"
    run_dir.mkdir(parents=False, exist_ok=False)
    (run_dir / "output").mkdir(parents=True, exist_ok=True)
    return run_dir


def fake_llm_mode_enabled() -> bool:
    return os.getenv("CSV_SANDBOX_AGENT_FAKE_LLM", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
