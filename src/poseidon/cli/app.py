"""Typer CLI entrypoint for Poseidon-CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from poseidon import APP_NAME, __version__
from poseidon.cli.banner import render_banner
from poseidon.config import RECOMMENDED_MODELS, Settings
from poseidon.core.single_file import FileExtractionError, SingleFileResult, run_single_file_request
from poseidon.llm.openrouter import OpenRouterClient, OpenRouterError
from poseidon.workspace import WorkspaceGuard, WorkspaceSecurityError

app = typer.Typer(
    name="poseidon",
    help=f"{APP_NAME} — Modular Python coding agent harness.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _select_model_interactive(default_model: str) -> str:
    """Let the developer pick a recommended model or enter a custom OpenRouter model ID."""
    table = Table(title="Select an OpenRouter Model", border_style="cyan")
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("Model ID", style="green")
    table.add_column("Notes", style="white")
    for idx, (model_id, desc) in enumerate(RECOMMENDED_MODELS, start=1):
        table.add_row(str(idx), model_id, desc)
    console.print(table)
    choice = Prompt.ask(
        "Choose a number or enter any OpenRouter model ID",
        default=default_model,
    ).strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(RECOMMENDED_MODELS):
            return RECOMMENDED_MODELS[idx][0]
    return choice or default_model


def _render_result(result: SingleFileResult, *, show_code: bool = True) -> None:
    action = "Updated" if result.is_update else "Created"
    console.print(f"\n[bold green]✓ {action}[/bold green] [yellow]{result.relative_path}[/yellow]")

    metrics = Table(show_header=False, box=None, padding=(0, 2))
    metrics.add_row("[bold]Model:[/bold]", result.model)
    metrics.add_row("[bold]Elapsed:[/bold]", f"{result.elapsed_seconds:.2f}s")
    out_tokens = (
        str(result.usage.completion_tokens)
        if result.usage.completion_tokens is not None
        else "unavailable"
    )
    total_tokens = (
        str(result.usage.total_tokens) if result.usage.total_tokens is not None else "unavailable"
    )
    cost = f"${result.usage.cost_usd:.6f}" if result.usage.cost_usd is not None else "unavailable"
    metrics.add_row("[bold]Output tokens:[/bold]", out_tokens)
    metrics.add_row("[bold]Total tokens:[/bold]", total_tokens)
    metrics.add_row("[bold]Cost (USD):[/bold]", cost)
    console.print(metrics)

    if show_code:
        syntax = Syntax(
            result.content,
            "python",
            theme="monokai",
            line_numbers=True,
        )
        console.print(syntax)


@app.command()
def version() -> None:
    """Print the Poseidon-CLI version."""
    console.print(f"[bold cyan]{APP_NAME}[/bold cyan] v{__version__}")


@app.command()
def models() -> None:
    """List recommended low-cost OpenRouter models."""
    table = Table(title=f"{APP_NAME} — Recommended OpenRouter Models", border_style="cyan")
    table.add_column("Model ID", style="green")
    table.add_column("Notes", style="white")
    for model_id, desc in RECOMMENDED_MODELS:
        table.add_row(model_id, desc)
    console.print(table)


@app.command()
def run(
    prompt: Annotated[
        str,
        typer.Argument(help="Coding request to execute in the workspace."),
    ],
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            "-w",
            help="Target workspace folder (the only directory Poseidon may read or modify).",
        ),
    ] = Path("."),
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="OpenRouter model ID (defaults to POSEIDON_DEFAULT_MODEL).",
        ),
    ] = None,
    target_file: Annotated[
        str | None,
        typer.Option(
            "--file",
            "-f",
            help="Optional relative filename inside the workspace to create or update.",
        ),
    ] = None,
    no_code: Annotated[
        bool,
        typer.Option(
            "--no-code",
            help="Suppress printing the full generated source code.",
        ),
    ] = False,
) -> None:
    """Execute a single coding request in the sandboxed workspace."""
    settings = Settings()
    chosen_model = model or settings.default_model

    try:
        guard = WorkspaceGuard(workspace)
    except WorkspaceSecurityError as exc:
        console.print(f"[bold red]Workspace error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    render_banner(console, model=chosen_model, workspace=guard.root)

    with OpenRouterClient(settings) as client:
        try:
            with console.status("[cyan]Calling OpenRouter...[/cyan]"):
                result = run_single_file_request(
                    client=client,
                    workspace=guard,
                    model=chosen_model,
                    prompt=prompt,
                    target_file=target_file,
                )
        except (OpenRouterError, FileExtractionError, WorkspaceSecurityError) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    _render_result(result, show_code=not no_code)


@app.command()
def chat(
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            "-w",
            help="Target workspace folder (the only directory Poseidon may read or modify).",
        ),
    ] = Path("."),
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="OpenRouter model ID. If omitted, Poseidon prompts you to pick one.",
        ),
    ] = None,
) -> None:
    """Start an interactive Poseidon-CLI session inside a workspace."""
    settings = Settings()

    try:
        guard = WorkspaceGuard(workspace)
    except WorkspaceSecurityError as exc:
        console.print(f"[bold red]Workspace error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    chosen_model = model or _select_model_interactive(settings.default_model)
    render_banner(console, model=chosen_model, workspace=guard.root)
    console.print(
        "[dim]Commands: /model (switch model), /files (list workspace files), /exit (quit)[/dim]\n"
    )

    with OpenRouterClient(settings) as client:
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]poseidon>[/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user_input:
                continue
            if user_input in {"/exit", "/quit", "exit", "quit"}:
                console.print("[dim]Goodbye.[/dim]")
                break
            if user_input == "/files":
                files = guard.list_files()
                console.print(
                    f"[yellow]Workspace files:[/yellow] {', '.join(files) if files else '(empty)'}"
                )
                continue
            if user_input == "/model":
                chosen_model = _select_model_interactive(chosen_model)
                console.print(f"[green]Active model switched to:[/green] {chosen_model}")
                continue

            try:
                with console.status(f"[cyan]Thinking ({chosen_model})...[/cyan]"):
                    result = run_single_file_request(
                        client=client,
                        workspace=guard,
                        model=chosen_model,
                        prompt=user_input,
                    )
                _render_result(result, show_code=True)
            except (OpenRouterError, FileExtractionError, WorkspaceSecurityError) as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")


def main() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    main()
