from __future__ import annotations

from rich.console import Console


console = Console()


def status_line(message: str) -> None:
    console.print(f"[bold cyan]csv-sandbox-agent[/bold cyan] {message}")
