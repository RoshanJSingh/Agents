from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from .config import SandboxConfig, fake_llm_mode_enabled
from .docker_runner import DockerRunner
from .llm_client import FakeLLMClient, OpenAICompatibleClient
from .logging_utils import console
from .orchestrator import CSVSandboxOrchestrator
from .schemas import DockerSandboxError, LLMResponseError


app = typer.Typer(help="Docker-isolated LLM-assisted CSV cleaning and baseline training.")


@app.command("profile")
def profile_command(
    input_csv: Path = typer.Option(..., "--input", help="Local CSV file to profile."),
    workspace: Path = typer.Option(..., "--workspace", help="Directory where run folders are created."),
) -> None:
    """Run deterministic Docker-based CSV profiling only."""

    orchestrator = CSVSandboxOrchestrator(llm_client=FakeLLMClient([]))
    try:
        run_dir = orchestrator.profile(input_csv=input_csv, workspace_root=workspace)
    except Exception as exc:
        console.print(f"[bold red]Profile failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    console.print("[bold green]Profile complete[/bold green]")
    console.print(f"Run directory: {run_dir}")
    console.print(f"Raw profile: {run_dir / 'raw_profile.json'}")


@app.command("run")
def run_command(
    input_csv: Path = typer.Option(..., "--input", help="Local CSV file to clean/train from."),
    workspace: Path = typer.Option(..., "--workspace", help="Directory where run folders are created."),
    target: Optional[str] = typer.Option(None, "--target", help="Optional target column for baseline training."),
    max_retries: int = typer.Option(5, "--max-retries", min=0, help="Debugger retry budget."),
    timeout_seconds: int = typer.Option(180, "--timeout-seconds", min=1, help="Per-attempt Docker timeout."),
    fake_llm: bool = typer.Option(False, "--fake-llm", help="Use a deterministic local fake LLM response."),
) -> None:
    """Run the full profiling, LLM generation, sandbox execution, and retry loop."""

    try:
        llm = _make_llm(fake_llm=fake_llm, target=target)
        orchestrator = CSVSandboxOrchestrator(llm_client=llm)
        result = orchestrator.run(
            input_csv=input_csv,
            workspace_root=workspace,
            target=target,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
    except (DockerSandboxError, LLMResponseError, FileNotFoundError, Exception) as exc:
        console.print(f"[bold red]Run failed before summary was written:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    _print_result(result.status, result.run_dir, result.attempts_used, result.message)
    if result.status == "success":
        console.print("[bold green]Outputs[/bold green]")
        for name, path in result.outputs.items():
            console.print(f"  {name}: {path}")
    else:
        console.print(f"[bold red]Last fingerprint:[/bold red] {result.last_failure_fingerprint}")
        console.print(f"[bold red]Last compacted error:[/bold red]\n{result.last_compacted_error or ''}")
        raise typer.Exit(1)


@app.command("doctor")
def doctor_command(
    fake_llm: bool = typer.Option(False, "--fake-llm", help="Do not require an LLM API key."),
) -> None:
    """Check dependencies, Docker reachability, sandbox image, and LLM configuration."""

    fake_mode = fake_llm or fake_llm_mode_enabled()
    table = Table(title="csv-sandbox-agent doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Details")

    failures = 0
    for module in [
        "docker",
        "pandas",
        "sklearn",
        "joblib",
        "pydantic",
        "typer",
        "rich",
        "pytest",
        "openai",
    ]:
        if importlib.util.find_spec(module):
            table.add_row(module, "[green]ok[/green]", "importable")
        else:
            failures += 1
            table.add_row(module, "[red]missing[/red]", "install project dependencies")

    docker_runner = DockerRunner(SandboxConfig())
    try:
        docker_runner.build_image_if_needed()
        table.add_row("Docker", "[green]ok[/green]", "reachable; sandbox image exists or was built")
    except Exception as exc:
        failures += 1
        table.add_row("Docker", "[red]failed[/red]", str(exc))

    if fake_mode:
        table.add_row("LLM key", "[yellow]skipped[/yellow]", "fake/mock mode enabled")
    elif os.getenv("OPENAI_API_KEY"):
        table.add_row("LLM key", "[green]ok[/green]", "OPENAI_API_KEY is set")
    else:
        failures += 1
        table.add_row(
            "LLM key",
            "[red]missing[/red]",
            "set OPENAI_API_KEY or run doctor with --fake-llm",
        )

    console.print(table)
    if failures:
        raise typer.Exit(1)


def _print_result(status: str, run_dir: Path, attempts_used: int, message: str) -> None:
    color = "green" if status == "success" else "red"
    console.print(f"[bold {color}]Status:[/bold {color}] {status}")
    console.print(f"Run directory: {run_dir}")
    console.print(f"Attempts used: {attempts_used}")
    console.print(f"Summary: {message}")


def _make_llm(fake_llm: bool, target: str | None) -> FakeLLMClient | OpenAICompatibleClient:
    if fake_llm or fake_llm_mode_enabled():
        profiler = {
            "dataset_summary": "Fake mode profiler report.",
            "likely_task_type": "unknown",
            "target_column": target,
            "column_issues": [],
            "global_recommendations": ["Clean null-like string values conservatively."],
            "warnings": ["Fake LLM mode skips real provider calls."],
        }
        return FakeLLMClient([json.dumps(profiler), _fake_cleaning_script()])
    return OpenAICompatibleClient()


def _fake_cleaning_script() -> str:
    return r'''
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
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    before_columns = [str(c) for c in df.columns]
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace({"": pd.NA, "na": pd.NA, "n/a": pd.NA, "null": pd.NA, "none": pd.NA, "nan": pd.NA, "-": pd.NA, "--": pd.NA, "?": pd.NA})
    df.to_csv(output_dir / "cleaned_data.csv", index=False)
    metrics = {
        "model_trained": False,
        "no_model_reason": "Fake LLM mode generated a cleaning-only script.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    manifest = {
        "input_path": str(input_path),
        "output_files": ["cleaned_data.csv", "metrics.json", "manifest.json"],
        "row_count_before": int(len(df)),
        "row_count_after": int(len(df)),
        "columns_before": before_columns,
        "columns_after": [str(c) for c in df.columns],
        "model_trained": False,
        "warnings": ["fake LLM mode"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
'''
