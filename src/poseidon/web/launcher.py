"""Launch and manage sibling Poseidon Web UI Docker containers from inside the CLI."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass

import docker
from docker.errors import APIError, DockerException, NotFound

DEFAULT_UI_IMAGE = "poseidon-cli:dev"
DEFAULT_CACHE_VOLUME = "poseidon-cli_poseidon_model_cache"
DEFAULT_SECRETS_VOLUME = "poseidon-cli_poseidon_secrets"
FORWARDED_ENV_KEYS: tuple[str, ...] = (
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_BASE_URL",
    "LMSTUDIO_BASE_URL",
    "RESPAN_API_KEY",
    "TAVILY_API_KEY",
    "BRAVE_API_KEY",
    "POSEIDON_DEFAULT_MODEL",
    "POSEIDON_SANDBOX_IMAGE",
    "POSEIDON_EMBEDDING_MODEL",
)


@dataclass(frozen=True)
class UIContainerInfo:
    """Metadata describing a running Poseidon Web UI Docker container."""

    name: str
    port: int
    url: str
    already_running: bool = False


def _discover_self_mounts(client: docker.DockerClient) -> tuple[str | None, str, str]:
    """Discover host `/workspace` source, image tag, and cache volume from the running container."""
    host_workspace = os.environ.get("POSEIDON_HOST_WORKSPACE") or None
    image_name = DEFAULT_UI_IMAGE
    cache_source = DEFAULT_CACHE_VOLUME

    try:
        me = client.containers.get(socket.gethostname())
        tags = getattr(me.image, "tags", None) or []
        if tags:
            image_name = str(tags[0])
        for mount in me.attrs.get("Mounts", []):
            dest = mount.get("Destination")
            if dest == "/workspace" and mount.get("Source"):
                host_workspace = str(mount["Source"])
            elif dest == "/home/poseidon/.cache/fastembed" and mount.get("Name"):
                cache_source = str(mount["Name"])
    except (DockerException, KeyError, TypeError):
        pass

    return host_workspace, image_name, cache_source


def launch_sibling_ui_container(
    *,
    port: int = 2701,
    model: str | None = None,
    plan_mode: bool = False,
    auto_approve: bool = False,
    resume_session_id: str | None = None,
    keep_alive: bool = False,
) -> UIContainerInfo:
    """Spawn a detached sibling `poseidon-ui-<port>` container sharing the current workspace."""
    client = docker.from_env()
    try:
        container_name = f"poseidon-ui-{port}"
        try:
            existing = client.containers.get(container_name)
            if existing.status == "running":
                return UIContainerInfo(
                    name=container_name,
                    port=port,
                    url=f"http://localhost:{port}",
                    already_running=True,
                )
        except NotFound:
            pass

        host_workspace, image_name, cache_volume = _discover_self_mounts(client)
        if not host_workspace:
            raise RuntimeError(
                "Could not determine host workspace path. Launch Poseidon via `poseidon` or "
                "set POSEIDON_HOST_WORKSPACE."
            )

        cmd: list[str] = [
            "ui",
            "--workspace",
            "/workspace",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]
        if model:
            cmd.extend(["--model", model])
        if plan_mode:
            cmd.append("--plan")
        if auto_approve:
            cmd.append("--yes")
        if resume_session_id:
            cmd.extend(["--resume", resume_session_id])
        if keep_alive:
            cmd.append("--keep-alive")

        env = {k: v for k in FORWARDED_ENV_KEYS if (v := os.environ.get(k))}
        env["POSEIDON_HOST_WORKSPACE"] = host_workspace
        volumes = {
            host_workspace: {"bind": "/workspace", "mode": "rw"},
            "/var/run/docker.sock": {"bind": "/var/run/docker.sock", "mode": "rw"},
            cache_volume: {"bind": "/home/poseidon/.cache/fastembed", "mode": "rw"},
            DEFAULT_SECRETS_VOLUME: {"bind": "/home/poseidon/.poseidon_secrets", "mode": "rw"},
        }

        client.containers.run(
            image=image_name,
            command=cmd,
            name=container_name,
            detach=True,
            auto_remove=True,
            working_dir="/workspace",
            ports={f"{port}/tcp": port},
            group_add=["0"],
            extra_hosts={"host.docker.internal": "host-gateway"},
            volumes=volumes,
            environment=env,
            labels={
                "poseidon.ui": "true",
                "poseidon.port": str(port),
                "poseidon.workspace": host_workspace,
            },
        )
        return UIContainerInfo(
            name=container_name,
            port=port,
            url=f"http://localhost:{port}",
            already_running=False,
        )
    except APIError as exc:
        raise RuntimeError(f"Docker failed to launch Web UI on port {port}: {exc}") from exc
    finally:
        client.close()


def stop_ui_container(port: int = 2701) -> bool:
    """Stop a running `poseidon-ui-<port>` (or default compose UI) container."""
    client = docker.from_env()
    try:
        candidates = [f"poseidon-ui-{port}"]
        if port == 2701:
            candidates.append("poseidon-cli-poseidon-ui-1")
        for name in candidates:
            try:
                container = client.containers.get(name)
                container.stop(timeout=5)
                return True
            except NotFound:
                continue
        return False
    finally:
        client.close()


def list_ui_containers() -> list[UIContainerInfo]:
    """List all currently running Poseidon Web UI containers."""
    client = docker.from_env()
    try:
        found: list[UIContainerInfo] = []
        seen: set[str] = set()
        for container in client.containers.list(filters={"label": "poseidon.ui=true"}):
            seen.add(container.name)
            port_str = container.labels.get("poseidon.port", "2701")
            port = int(port_str) if port_str.isdigit() else 2701
            found.append(
                UIContainerInfo(
                    name=container.name,
                    port=port,
                    url=f"http://localhost:{port}",
                    already_running=True,
                )
            )
        for container in client.containers.list(filters={"name": "poseidon-cli-poseidon-ui"}):
            if container.name not in seen:
                found.append(
                    UIContainerInfo(
                        name=container.name,
                        port=2701,
                        url="http://localhost:2701",
                        already_running=True,
                    )
                )
        return found
    finally:
        client.close()
