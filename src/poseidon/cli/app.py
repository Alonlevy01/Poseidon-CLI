"""Typer CLI entrypoint for Poseidon-CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from poseidon import APP_NAME, __version__
from poseidon.cli.banner import render_banner
from poseidon.config import RECOMMENDED_MODELS, Settings
from poseidon.core.benchmark_m2 import (
    DEFAULT_BENCHMARK_PROMPT,
    BenchmarkRow,
    run_whole_vs_diff_benchmark,
)
from poseidon.core.diff_edit import DiffEditResult, run_diff_edit_request
from poseidon.core.single_file import FileExtractionError, SingleFileResult, run_single_file_request
from poseidon.llm.openrouter import OpenRouterClient, OpenRouterError
from poseidon.patching.unified_diff import PatchError, apply_unified_diff
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


def _format_metric(value: int | None) -> str:
    return str(value) if value is not None else "unavailable"


def _format_cost(cost: float | None) -> str:
    return f"${cost:.6f}" if cost is not None else "unavailable"


def _render_single_file_result(result: SingleFileResult, *, show_code: bool = True) -> None:
    action = "Updated" if result.is_update else "Created"
    console.print(f"\n[bold green]✓ {action}[/bold green] [yellow]{result.relative_path}[/yellow]")

    metrics = Table(show_header=False, box=None, padding=(0, 2))
    metrics.add_row("[bold]Model:[/bold]", result.model)
    metrics.add_row("[bold]Elapsed:[/bold]", f"{result.elapsed_seconds:.2f}s")
    metrics.add_row("[bold]Output tokens:[/bold]", _format_metric(result.usage.completion_tokens))
    metrics.add_row("[bold]Total tokens:[/bold]", _format_metric(result.usage.total_tokens))
    metrics.add_row("[bold]Cost (USD):[/bold]", _format_cost(result.usage.cost_usd))
    console.print(metrics)

    if show_code:
        syntax = Syntax(result.content, "python", theme="monokai", line_numbers=True)
        console.print(syntax)


def _preview_diff(diff_text: str, *, auto_approve: bool) -> bool:
    """Display a syntax-highlighted unified diff preview and optionally confirm before applying."""
    console.print(
        Panel(
            Syntax(diff_text, "diff", theme="monokai", line_numbers=False),
            title="[bold cyan]Unified Diff Preview[/bold cyan]",
            border_style="cyan",
        )
    )
    if auto_approve:
        return True
    return Confirm.ask("Apply this patch?", default=True)


def _run_diff_with_spinner(
    *,
    client: OpenRouterClient,
    workspace: WorkspaceGuard,
    model: str,
    prompt: str,
    target_file: str,
    auto_approve: bool,
) -> DiffEditResult:
    """Run diff generation with a spinner that stops cleanly before interactive preview/approval."""
    status = console.status(f"[cyan]Generating unified diff ({model})...[/cyan]")
    status.start()
    try:

        def _confirm(diff_text: str) -> bool:
            status.stop()
            return _preview_diff(diff_text, auto_approve=auto_approve)

        return run_diff_edit_request(
            client=client,
            workspace=workspace,
            model=model,
            prompt=prompt,
            target_file=target_file,
            preview_and_confirm=_confirm,
        )
    finally:
        status.stop()


def _render_diff_result(result: DiffEditResult) -> None:
    if result.applied:
        files_str = ", ".join(result.modified_files)
        console.print(
            f"\n[bold green]✓ Patched[/bold green] [yellow]{files_str}[/yellow] "
            f"([green]+{result.added_lines}[/green] / [red]-{result.removed_lines}[/red])"
        )
    elif result.rejected_by_user:
        console.print("\n[bold yellow]⊘ Patch skipped by user (files unchanged).[/bold yellow]")
    else:
        console.print(f"\n[bold red]✗ Patch failed (files unchanged):[/bold red] {result.reason}")

    metrics = Table(show_header=False, box=None, padding=(0, 2))
    metrics.add_row("[bold]Model:[/bold]", result.model)
    metrics.add_row("[bold]Attempts:[/bold]", str(result.attempts))
    metrics.add_row("[bold]Elapsed:[/bold]", f"{result.elapsed_seconds:.2f}s")
    metrics.add_row("[bold]Output tokens:[/bold]", _format_metric(result.total_output_tokens))
    metrics.add_row("[bold]Total tokens:[/bold]", _format_metric(result.total_tokens))
    metrics.add_row("[bold]Cost (USD):[/bold]", _format_cost(result.total_cost_usd))
    console.print(metrics)


def _render_benchmark_table(rows: list[BenchmarkRow]) -> None:
    table = Table(
        title="Milestone 2 Benchmark — Whole-File vs. Unified Diff",
        border_style="bright_blue",
    )
    table.add_column("Strategy", style="bold cyan")
    table.add_column("Edit Applied", style="white")
    table.add_column("Test Result", style="white")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Elapsed (incl. retries)", justify="right")
    table.add_column("Cost (USD)", justify="right")

    for row in rows:
        applied_cell = "[green]Yes[/green]" if row.edit_applied else "[red]No[/red]"
        test_cell = (
            f"[bold green]{row.test_detail}[/bold green]"
            if row.test_passed
            else f"[bold red]{row.test_detail}[/bold red]"
        )
        table.add_row(
            f"{row.strategy} ({row.attempts}x)",
            applied_cell,
            test_cell,
            _format_metric(row.total_output_tokens),
            f"{row.elapsed_seconds:.2f}s",
            _format_cost(row.cost_usd),
        )
    console.print(table)


@app.command()
def version() -> None:
    """Print the Poseidon-CLI version."""
    console.print(f"[bold cyan]{APP_NAME}[/bold cyan] v{__version__}")


@app.command()
def models() -> None:
    """List recommended free OpenRouter models."""
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
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Edit mode: 'auto' (diff if file exists, else whole), 'diff', or 'whole'.",
        ),
    ] = "auto",
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Auto-approve diff patches without prompting."),
    ] = False,
    no_code: Annotated[
        bool,
        typer.Option("--no-code", help="Suppress printing full source code in whole-file mode."),
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

    existing_files = guard.list_files()
    resolved_target = target_file or (existing_files[0] if len(existing_files) == 1 else None)
    use_diff = mode == "diff" or (
        mode == "auto" and resolved_target is not None and resolved_target in existing_files
    )

    with OpenRouterClient(settings) as client:
        try:
            if use_diff:
                if not resolved_target:
                    console.print("[bold red]Error:[/bold red] Specify --file for diff mode.")
                    raise typer.Exit(code=1)
                diff_res = _run_diff_with_spinner(
                    client=client,
                    workspace=guard,
                    model=chosen_model,
                    prompt=prompt,
                    target_file=resolved_target,
                    auto_approve=yes,
                )
                _render_diff_result(diff_res)
                if not diff_res.applied and not diff_res.rejected_by_user:
                    raise typer.Exit(code=1)
            else:
                with console.status(f"[cyan]Calling OpenRouter ({chosen_model})...[/cyan]"):
                    result = run_single_file_request(
                        client=client,
                        workspace=guard,
                        model=chosen_model,
                        prompt=prompt,
                        target_file=target_file,
                    )
                _render_single_file_result(result, show_code=not no_code)
        except (OpenRouterError, FileExtractionError, WorkspaceSecurityError) as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc


@app.command(name="apply-patch")
def apply_patch(
    patch_file: Annotated[
        Path,
        typer.Argument(help="Path to a unified diff file to validate and apply."),
    ],
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Target workspace folder."),
    ] = Path("."),
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip interactive confirmation."),
    ] = False,
) -> None:
    """Preview, validate, and atomically apply a unified diff file to the workspace."""
    try:
        guard = WorkspaceGuard(workspace)
        diff_text = patch_file.read_text(encoding="utf-8")
    except (WorkspaceSecurityError, OSError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not _preview_diff(diff_text, auto_approve=yes):
        console.print("[yellow]Patch cancelled (files unchanged).[/yellow]")
        return

    try:
        result = apply_unified_diff(workspace=guard, diff_text=diff_text)
    except (PatchError, WorkspaceSecurityError) as exc:
        console.print(f"[bold red]✗ Patch rejected (files left intact):[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold green]✓ Applied {result.hunks_applied} hunk(s)[/bold green] to "
        f"[yellow]{', '.join(result.modified_files)}[/yellow] "
        f"([green]+{result.added_lines}[/green] / [red]-{result.removed_lines}[/red])"
    )


@app.command(name="benchmark-diff")
def benchmark_diff(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Workspace containing the demo file."),
    ] = Path("."),
    target_file: Annotated[
        str,
        typer.Option("--file", "-f", help="Target demo file inside workspace."),
    ] = "ledger.py",
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="OpenRouter model ID to benchmark."),
    ] = None,
    prompt: Annotated[
        str,
        typer.Option("--prompt", "-p", help="Small edit request to compare."),
    ] = DEFAULT_BENCHMARK_PROMPT,
) -> None:
    """Compare whole-file vs. unified-diff edits from identical copies of the demo file."""
    settings = Settings()
    chosen_model = model or settings.default_model
    guard = WorkspaceGuard(workspace)
    render_banner(console, model=chosen_model, workspace=guard.root)

    console.print(f"[bold]Benchmark target:[/bold] [yellow]{target_file}[/yellow]")
    console.print(f"[bold]Request:[/bold] {prompt}\n")

    with OpenRouterClient(settings) as client:
        try:
            spinner_msg = f"[cyan]Running whole-file vs. diff benchmark ({chosen_model})...[/cyan]"
            with console.status(spinner_msg):
                rows = run_whole_vs_diff_benchmark(
                    client=client,
                    source_workspace=guard.root,
                    target_file=target_file,
                    model=chosen_model,
                    prompt=prompt,
                )
        except (FileNotFoundError, OpenRouterError) as exc:
            console.print(f"[bold red]Benchmark error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    for row in rows:
        if row.strategy == "unified-diff" and row.diff_text:
            console.print(
                Panel(
                    Syntax(row.diff_text, "diff", theme="monokai"),
                    title="[bold cyan]Generated Unified Diff[/bold cyan]",
                    border_style="cyan",
                )
            )
    _render_benchmark_table(rows)


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
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Edit mode: 'auto' (diff for existing files, whole for new), 'diff', or 'whole'.",
        ),
    ] = "auto",
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Auto-approve diff patches without prompting."),
    ] = False,
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
        f"[dim]Mode: {mode} | Commands: /model, /mode <auto|diff|whole>, /files, /exit[/dim]\n"
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
            if user_input.startswith("/mode"):
                parts = user_input.split()
                if len(parts) == 2 and parts[1] in {"auto", "diff", "whole"}:
                    mode = parts[1]
                    console.print(f"[green]Edit mode switched to:[/green] {mode}")
                else:
                    console.print("[yellow]Usage: /mode <auto|diff|whole>[/yellow]")
                continue

            existing_files = guard.list_files()
            use_diff = mode == "diff" or (mode == "auto" and len(existing_files) == 1)

            try:
                if use_diff and existing_files:
                    target_file = existing_files[0]
                    diff_res = _run_diff_with_spinner(
                        client=client,
                        workspace=guard,
                        model=chosen_model,
                        prompt=user_input,
                        target_file=target_file,
                        auto_approve=yes,
                    )
                    _render_diff_result(diff_res)
                else:
                    with console.status(f"[cyan]Thinking ({chosen_model})...[/cyan]"):
                        result = run_single_file_request(
                            client=client,
                            workspace=guard,
                            model=chosen_model,
                            prompt=user_input,
                        )
                    _render_single_file_result(result, show_code=True)
            except (OpenRouterError, FileExtractionError, WorkspaceSecurityError) as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")


def main() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    main()
