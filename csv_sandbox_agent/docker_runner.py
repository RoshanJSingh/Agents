from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .config import SandboxConfig
from .profiling import PROFILER_SCRIPT
from .schemas import DockerRunResult, DockerSandboxError


class DockerRunner:
    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import docker
            except Exception as exc:
                raise DockerSandboxError(
                    "Docker Python SDK is not installed. Install project dependencies."
                ) from exc
            try:
                self._client = docker.from_env()
                self._client.ping()
            except Exception as exc:
                raise DockerSandboxError("Docker is not reachable") from exc
        return self._client

    def build_image_if_needed(self) -> None:
        try:
            self.client.images.get(self.config.image_name)
            return
        except Exception:
            pass

        dockerfile = self.config.dockerfile_path
        if not dockerfile.exists():
            raise DockerSandboxError(f"Sandbox Dockerfile not found: {dockerfile}")
        try:
            self.client.images.build(
                path=str(dockerfile.parent),
                dockerfile=dockerfile.name,
                tag=self.config.image_name,
                rm=True,
            )
        except Exception as exc:
            raise DockerSandboxError(f"Failed to build sandbox image: {exc}") from exc

    def run_profiler(self, workspace_path: Path) -> DockerRunResult:
        return self._run(
            workspace_path=workspace_path,
            command=["python", "-c", PROFILER_SCRIPT],
            timeout_seconds=self.config.profiler_timeout_seconds,
        )

    def run_generated_script(
        self,
        workspace_path: Path,
        target: str | None,
        timeout_seconds: int,
    ) -> DockerRunResult:
        return self._run(
            workspace_path=workspace_path,
            command=[
                "python",
                "/workspace/generated_script.py",
                "--input",
                "/workspace/input.csv",
                "--output-dir",
                "/workspace/output",
                "--target",
                target or "",
            ],
            timeout_seconds=timeout_seconds,
        )

    def _run(
        self,
        *,
        workspace_path: Path,
        command: list[str],
        timeout_seconds: int,
    ) -> DockerRunResult:
        workspace = workspace_path.resolve()
        if not workspace.exists() or not workspace.is_dir():
            raise DockerSandboxError(f"Workspace directory does not exist: {workspace}")

        container = None
        start = time.monotonic()
        timed_out = False
        exit_code = 125
        stdout = ""
        stderr = ""
        try:
            container = self.client.containers.run(
                self.config.image_name,
                command=command,
                detach=True,
                network_disabled=True,
                working_dir="/workspace",
                user=_container_user(),
                volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
                mem_limit=self.config.memory_limit,
                nano_cpus=self.config.nano_cpus,
                pids_limit=self.config.pids_limit,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                read_only=True,
                tmpfs={"/tmp": "rw,noexec,nosuid,size=64m"},
            )
            while True:
                container.reload()
                if container.status == "exited":
                    result = container.wait()
                    exit_code = int(result.get("StatusCode", 125))
                    break
                if time.monotonic() - start > timeout_seconds:
                    timed_out = True
                    try:
                        container.kill()
                    except Exception:
                        pass
                    exit_code = 124
                    break
                time.sleep(0.2)
            stdout = _decode_logs(container.logs(stdout=True, stderr=False))
            stderr = _decode_logs(container.logs(stdout=False, stderr=True))
            if timed_out:
                stderr = (stderr + "\n" if stderr else "") + (
                    f"Docker execution timed out after {timeout_seconds} seconds"
                )
        except Exception as exc:
            raise DockerSandboxError(f"Docker execution failed: {exc}") from exc
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
        return DockerRunResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            duration_seconds=time.monotonic() - start,
        )


def _decode_logs(logs: bytes | str) -> str:
    if isinstance(logs, bytes):
        return logs.decode("utf-8", errors="replace")
    return logs


def _container_user() -> str:
    """Run as the host UID/GID on POSIX so the mounted workspace remains writable."""

    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        uid = os.getuid()
        gid = os.getgid()
        if uid != 0:
            return f"{uid}:{gid}"
    return "sandbox"
