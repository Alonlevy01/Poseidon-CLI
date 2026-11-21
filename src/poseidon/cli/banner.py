"""Terminal banner and Rich display helpers for Poseidon-CLI."""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from poseidon import APP_NAME, __version__

TRIDENT_ART = r"""
    ♆   ____   ___  ____  _____ ___ ____   ___  _   _        ____ _     ___
       |  _ \ / _ \/ ___|| ____|_ _|  _ \ / _ \| \ | |      / ___| |   |_ _|
       | |_) | | | \___ \|  _|  | || | | | | | |  \| |_____| |   | |    | |
       |  __/| |_| |___) | |___ | || |_| | |_| | |\  |_____| |___| |___ | |
       |_|    \___/|____/|_____|___|____/ \___/|_| \_|      \____|_____|___|
"""


def render_banner(console: Console, *, model: str, workspace: Path) -> None:
    """Display the Poseidon-CLI banner with active model and sandboxed workspace."""
    title = Text(f"{APP_NAME} v{__version__}", style="bold cyan")
    body = Text(TRIDENT_ART.strip("\n"), style="cyan")
    body.append("\n\n")
    body.append("Model:     ", style="bold white")
    body.append(f"{model}\n", style="green")
    body.append("Workspace: ", style="bold white")
    body.append(f"{workspace}", style="yellow")
    console.print(Panel(body, title=title, border_style="bright_blue", expand=False))
