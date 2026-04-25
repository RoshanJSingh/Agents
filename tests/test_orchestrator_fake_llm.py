import json
from pathlib import Path

from csv_sandbox_agent.llm_client import FakeLLMClient
from csv_sandbox_agent.orchestrator import CSVSandboxOrchestrator
from csv_sandbox_agent.schemas import DockerRunResult


BAD_SCRIPT = r'''
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="")
    args = parser.parse_args()
    Path("HOST_EXECUTED").write_text("bad", encoding="utf-8")
    df = pd.read_csv(args.input)
    raise ValueError("simulated generated script failure")
    df.to_csv(Path(args.output_dir) / "cleaned_data.csv", index=False)
    (Path(args.output_dir) / "metrics.json").write_text(json.dumps({"model_trained": False}), encoding="utf-8")
    (Path(args.output_dir) / "manifest.json").write_text(json.dumps({"model_trained": False}), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


FIXED_SCRIPT = r'''
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", default="")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    df.to_csv(output_dir / "cleaned_data.csv", index=False)
    metrics = {"model_trained": False, "no_model_reason": "test fake runner"}
    (output_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    manifest = {
        "input_path": str(args.input),
        "output_files": ["cleaned_data.csv", "metrics.json", "manifest.json"],
        "row_count_before": int(len(df)),
        "row_count_after": int(len(df)),
        "columns_before": [str(c) for c in df.columns],
        "columns_after": [str(c) for c in df.columns],
        "model_trained": False,
        "warnings": [],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


class FakeDockerRunner:
    def __init__(self) -> None:
        self.run_count = 0
        self.built = False

    def build_image_if_needed(self) -> None:
        self.built = True

    def run_profiler(self, workspace_path: Path) -> DockerRunResult:
        profile = {
            "columns": ["feature", "target"],
            "columns_profile": {
                "feature": {"dtype": "object", "missing_count": 0, "sample_values": ["a"]},
                "target": {"dtype": "object", "missing_count": 0, "sample_values": ["yes"]},
            },
            "warnings": [],
            "read_errors": [],
        }
        (workspace_path / "raw_profile.json").write_text(json.dumps(profile), encoding="utf-8")
        return DockerRunResult(exit_code=0, stdout="profile ok", stderr="", timed_out=False, duration_seconds=0.1)

    def run_generated_script(self, workspace_path: Path, target: str | None, timeout_seconds: int) -> DockerRunResult:
        self.run_count += 1
        if self.run_count == 1:
            return DockerRunResult(
                exit_code=1,
                stdout="",
                stderr='Traceback (most recent call last):\n  File "/workspace/generated_script.py", line 17, in main\nValueError: simulated generated script failure\n',
                timed_out=False,
                duration_seconds=0.2,
            )
        output_dir = workspace_path / "output"
        output_dir.mkdir(exist_ok=True)
        (output_dir / "cleaned_data.csv").write_text("feature,target\na,yes\n", encoding="utf-8")
        (output_dir / "metrics.json").write_text(
            json.dumps({"model_trained": False, "no_model_reason": "fake success"}),
            encoding="utf-8",
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "input_path": "/workspace/input.csv",
                    "output_files": ["cleaned_data.csv", "metrics.json", "manifest.json"],
                    "row_count_before": 1,
                    "row_count_after": 1,
                    "columns_before": ["feature", "target"],
                    "columns_after": ["feature", "target"],
                    "model_trained": False,
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )
        return DockerRunResult(exit_code=0, stdout="ok", stderr="", timed_out=False, duration_seconds=0.2)


def test_orchestrator_fake_llm_retries_once_and_succeeds(tmp_path):
    input_csv = tmp_path / "messy.csv"
    input_csv.write_text("feature,target\na,yes\n", encoding="utf-8")
    profiler_report = {
        "dataset_summary": "test",
        "likely_task_type": "classification",
        "target_column": "target",
        "column_issues": [],
        "global_recommendations": [],
        "warnings": [],
    }
    llm = FakeLLMClient([json.dumps(profiler_report), BAD_SCRIPT, FIXED_SCRIPT])
    fake_docker = FakeDockerRunner()
    orchestrator = CSVSandboxOrchestrator(llm_client=llm, docker_runner=fake_docker)  # type: ignore[arg-type]

    result = orchestrator.run(
        input_csv=input_csv,
        workspace_root=tmp_path / "runs",
        target="target",
        max_retries=5,
        timeout_seconds=5,
    )

    assert result.status == "success"
    assert result.attempts_used == 2
    assert fake_docker.built is True
    assert fake_docker.run_count == 2
    assert (result.run_dir / "generated_script_attempt_0.py").exists()
    assert (result.run_dir / "generated_script_attempt_1.py").exists()
    assert (result.run_dir / "output" / "cleaned_data.csv").exists()
    assert not (Path.cwd() / "HOST_EXECUTED").exists()
    assert not (result.run_dir / "HOST_EXECUTED").exists()
