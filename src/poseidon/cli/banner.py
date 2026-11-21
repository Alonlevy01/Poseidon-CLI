"""Terminal banner and Rich display helpers for Poseidon-CLI."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from poseidon import APP_NAME, __version__

BANNER_ROWS: list[tuple[str, str]] = [
    (
        r"    /\   /\   /\    ",
        r"  ____   ___  ____  _____ ___ ____   ___  _   _        ____ _     ___ ",
    ),
    (
        r"    ||   ||   ||    ",
        r" |  _ \ / _ \/ ___|| ____|_ _|  _ \ / _ \| \ | |      / ___| |   |_ _|",
    ),
    (
        r"    \\___||___//    ",
        r" | |_) | | | \___ \|  _|  | || | | | | | |  \| |_____| |   | |    | | ",
    ),
    (
        r"     \___||___/     ",
        r" |  __/| |_| |___) | |___ | || |_| | |_| | |\  |_____| |___| |___ | | ",
    ),
    (
        r"         ||         ",
        r" |_|    \___/|____/|_____|___|____/ \___/|_| \_|      \____|_____|___|",
    ),
    (
        r"         ||         ",
        r"                                                                      ",
    ),
]


def render_banner(
    console: Console,
    *,
    model: str,
    workspace: Path,
    plan_mode: bool = False,
    memory_sources: list[str] | None = None,
    session_id: str | None = None,
) -> None:
    """Display the Poseidon-CLI banner with gold trident, ocean-cyan title, and active context."""
    title = Text(f"{APP_NAME} v{__version__}", style="bold cyan")
    body = Text()
    for trident_part, logo_part in BANNER_ROWS:
        body.append(trident_part, style="bold bright_yellow")
        body.append(logo_part, style="bold cyan")
        body.append("\n")

    body.append("\n")
    body.append("Model:     ", style="bold white")
    body.append(f"{model}\n", style="green")
    body.append("Workspace: ", style="bold white")
    body.append(f"{workspace}", style="yellow")
    if plan_mode:
        body.append("\nMode:      ", style="bold white")
        body.append("PLAN (read-only)", style="bold magenta")
    if memory_sources:
        body.append("\nMemory:    ", style="bold white")
        body.append(", ".join(memory_sources), style="cyan")
    if session_id:
        body.append("\nSession:   ", style="bold white")
        body.append(session_id, style="dim")

    console.print(Panel(body, title=title, border_style="bright_blue", expand=False))
