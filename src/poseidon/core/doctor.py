"""`poseidon doctor` — 1-second environment, Docker, sandbox, provider & local AI health check."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import docker
from docker.errors import DockerException, ImageNotFound
from pydantic import BaseModel

from poseidon.config import Settings
from poseidon.context.memory import load_project_memory
from poseidon.context.secrets_store import ProviderSecretsStore
from poseidon.context.user_settings import UserSettingsStore, list_mcp_configs
from poseidon.llm.openrouter import discover_local_models


class DoctorCheck(BaseModel):
    """Result for one diagnostic check in `poseidon doctor`."""

    name: str
    ok: bool
    status: str
    detail: str


def _binary_version(cmd: list[str]) -> str | None:
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=4, check=False)
        first = (proc.stdout or proc.stderr).strip().splitlines()
        return first[0][:80] if first else "installed"
    except (OSError, subprocess.SubprocessError):
        return "installed"


def run_doctor_checks(
    *,
    workspace_root: Path,
    settings: Settings | None = None,
    secrets_store: ProviderSecretsStore | None = None,
) -> list[DoctorCheck]:
    """Run all environment, Docker, tool, provider, and local AI health checks."""
    cfg = settings or Settings()
    user_cfg = UserSettingsStore(workspace_root).load()
    secrets = secrets_store or ProviderSecretsStore()
    sandbox_img = user_cfg.sandbox_image or cfg.sandbox_image
    checks: list[DoctorCheck] = []

    # 1. Docker daemon + sandbox image
    try:
        dclient = docker.from_env()
        try:
            ver = dclient.version().get("Version", "ok")
            checks.append(
                DoctorCheck(
                    name="Docker Daemon",
                    ok=True,
                    status="OK",
                    detail=f"Connected (engine {ver})",
                )
            )
            try:
                dclient.images.get(sandbox_img)
                checks.append(
                    DoctorCheck(
                        name="Sandbox Image",
                        ok=True,
                        status="OK",
                        detail=f"{sandbox_img} present locally",
                    )
                )
            except ImageNotFound:
                checks.append(
                    DoctorCheck(
                        name="Sandbox Image",
                        ok=False,
                        status="MISSING",
                        detail=(
                            f"{sandbox_img} not built "
                            "(run docker compose --profile build-only build)"
                        ),
                    )
                )
        finally:
            dclient.close()
    except DockerException as exc:
        checks.append(
            DoctorCheck(
                name="Docker Daemon",
                ok=False,
                status="ERROR",
                detail=f"Cannot reach Docker socket: {exc}",
            )
        )

    # 2. ripgrep & git binaries
    rg_ver = _binary_version(["rg", "--version"])
    checks.append(
        DoctorCheck(
            name="ripgrep (rg)",
            ok=rg_ver is not None,
            status="OK" if rg_ver else "MISSING",
            detail=rg_ver or "Not installed (Python fallback active)",
        )
    )
    git_ver = _binary_version(["git", "--version"])
    checks.append(
        DoctorCheck(
            name="Git CLI",
            ok=git_ver is not None,
            status="OK" if git_ver else "MISSING",
            detail=git_ver or "Not installed (git_status/git_diff unavailable)",
        )
    )

    # 3. Cloud API Keys (masked, never plaintext)
    key_status = secrets.masked_status(cfg)
    configured_cloud = [
        f"{v['label']} (keyring: {v['keyring_target']})"
        if v.get("keyring_target")
        else f"{v['label']} ({v['source']})"
        for v in key_status.values()
        if v["configured"]
    ]
    checks.append(
        DoctorCheck(
            name="Cloud API Keys",
            ok=bool(configured_cloud),
            status="OK" if configured_cloud else "WARN",
            detail=", ".join(configured_cloud) if configured_cloud else "No cloud API keys set",
        )
    )

    # 4. Local AI (Ollama & LM Studio)
    local_info = discover_local_models(
        ollama_base_url=user_cfg.ollama_base_url or cfg.ollama_base_url,
        lmstudio_base_url=user_cfg.lmstudio_base_url or cfg.lmstudio_base_url,
        timeout_seconds=1.5,
    )
    for prov, label in (("ollama", "Ollama (Local AI)"), ("lmstudio", "LM Studio (Local AI)")):
        li = local_info[prov]
        checks.append(
            DoctorCheck(
                name=label,
                ok=bool(li["reachable"]),
                status="ONLINE" if li["reachable"] else "OFFLINE",
                detail=(
                    f"{len(li['models'])} model(s) at {li['base_url']}"
                    if li["reachable"]
                    else f"Not running at {li['base_url']}"
                ),
            )
        )

    # 5. Workspace Memory & MCP
    mem = load_project_memory(workspace_root)
    checks.append(
        DoctorCheck(
            name="Project Memory",
            ok=bool(mem.sources),
            status="LOADED" if mem.sources else "NONE",
            detail=(
                f"{', '.join(mem.sources)} ({len(mem.content)} chars)"
                if mem.sources
                else "Run `/init` to create POSEIDON.md"
            ),
        )
    )
    mcp_cfgs = list_mcp_configs(workspace_root)
    checks.append(
        DoctorCheck(
            name="MCP Servers",
            ok=True,
            status=f"{len(mcp_cfgs)} CONFIGURED",
            detail=(
                ", ".join(mcp_cfgs.keys())
                if mcp_cfgs
                else "None configured (.poseidon/mcp.json)"
            ),
        )
    )
    return checks
