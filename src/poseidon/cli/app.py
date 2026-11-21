"""Typer CLI entrypoint for Poseidon-CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from poseidon import APP_NAME, __version__
from poseidon.cli.banner import render_banner
from poseidon.config import RECOMMENDED_MODELS, Settings
from poseidon.context.checkpoints import WorkspaceCheckpointManager
from poseidon.context.compactor import compact_messages
from poseidon.context.memory import load_project_memory, starter_poseidon_md
from poseidon.context.sessions import SessionStore, generate_session_id
from poseidon.core.agent import AgentRunResult, ToolEvent, build_system_prompt, run_agent_loop
from poseidon.core.benchmark_challenge import (
    ChallengeRunRow,
    run_optional_challenge_benchmark,
)
from poseidon.core.benchmark_m2 import (
    DEFAULT_BENCHMARK_PROMPT,
    BenchmarkRow,
    run_whole_vs_diff_benchmark,
)
from poseidon.core.benchmark_m6 import SearchBenchmarkReport, run_search_benchmark
from poseidon.core.benchmark_m7 import (
    DEFAULT_M5_REQUEST,
    ModelComparisonRow,
    run_model_comparison,
)
from poseidon.core.diff_edit import DiffEditResult, run_diff_edit_request
from poseidon.core.single_file import FileExtractionError, SingleFileResult, run_single_file_request
from poseidon.indexing.semantic_index import FastEmbedProvider, SemanticCodeIndex
from poseidon.llm.openrouter import (
    ModelToolSupportError,
    OpenRouterClient,
    OpenRouterError,
    TokenUsage,
)
from poseidon.mcp.client import MCPClientManager
from poseidon.observability.respan import PoseidonTracer
from poseidon.patching.unified_diff import PatchError, apply_unified_diff
from poseidon.plugins.hooks import HookManager
from poseidon.plugins.loader import load_workspace_plugins
from poseidon.sandbox.docker_runner import DockerTestRunner, TestRunResult
from poseidon.tools.builtin import register_core_tools
from poseidon.tools.registry import ToolRegistry, ToolSessionState
from poseidon.workspace import WorkspaceGuard, WorkspaceSecurityError

app = typer.Typer(
    name="poseidon",
    help=f"{APP_NAME} — Modular Python coding agent harness. Run bare `poseidon` to chat.",
    no_args_is_help=False,
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


def _validate_or_reselect_model(
    client: OpenRouterClient,
    model_id: str,
    *,
    interactive: bool,
) -> str:
    """Validate that a model exists and supports tool calling; allow re-selection if not."""
    current = model_id
    while True:
        try:
            client.validate_tool_model(current)
            return current
        except ModelToolSupportError as exc:
            console.print(f"[bold red]Model check failed:[/bold red] {exc}")
            if not interactive:
                console.print(
                    "[yellow]Run `poseidon models` to pick a tool-capable model.[/yellow]"
                )
                raise typer.Exit(code=1) from exc
            console.print("[yellow]Please choose another tool-capable model:[/yellow]")
            current = _select_model_interactive(RECOMMENDED_MODELS[0][0])
        except OpenRouterError as exc:
            # If catalog check fails due to transient network error, warn and continue
            console.print(f"[yellow]Warning: could not pre-check model catalog ({exc}).[/yellow]")
            return current


def _format_metric(value: int | None) -> str:
    return str(value) if value is not None else "unavailable"


def _format_output_tokens(usage: TokenUsage) -> str:
    if usage.completion_tokens is None:
        return "unavailable"
    if usage.reasoning_tokens:
        return f"{usage.completion_tokens} ({usage.reasoning_tokens} reasoning)"
    return str(usage.completion_tokens)


def _format_cost(cost: float | None) -> str:
    return f"${cost:.6f}" if cost is not None else "unavailable"


def _render_test_result(result: TestRunResult) -> None:
    """Display structured isolated test-runner output in the terminal."""
    border = "green" if result.verified else "red"
    status_badge = (
        "[bold green]VERIFIED[/bold green]"
        if result.verified
        else "[bold red]UNVERIFIED[/bold red]"
    )

    lines = [
        f"[bold]Command:[/bold]   {result.command}",
        "[bold]Sandbox:[/bold]   disposable Docker (--network none | zero secrets)",
        f"[bold]Timeout:[/bold]   {result.timeout_seconds}s  |  "
        f"[bold]Exit:[/bold] {result.exit_code}  |  "
        f"[bold]Elapsed:[/bold] {result.elapsed_seconds:.2f}s",
        f"[bold]Results:[/bold]   [green]{result.passed} passed[/green], "
        f"[red]{result.failed} failed[/red], "
        f"[yellow]{result.errors} errors[/yellow], "
        f"{result.skipped} skipped (total {result.total})",
        f"[bold]Status:[/bold]    {status_badge}",
    ]
    if result.unverified_reason:
        lines.append(f"[bold red]Reason:[/bold red]    {result.unverified_reason}")

    if result.failures:
        lines.append("\n[bold red]Failing Tests:[/bold red]")
        for fail in result.failures:
            lines.append(f"  [red]✗ {fail.name}[/red]")
            lines.append(f"    {fail.message}")

    console.print(
        Panel(
            "\n".join(lines),
            title="[bold cyan]Isolated Unit Test Report[/bold cyan]",
            border_style=border,
        )
    )


def _render_single_file_result(result: SingleFileResult, *, show_code: bool = True) -> None:
    action = "Updated" if result.is_update else "Created"
    console.print(f"\n[bold green]✓ {action}[/bold green] [yellow]{result.relative_path}[/yellow]")

    metrics = Table(show_header=False, box=None, padding=(0, 2))
    metrics.add_row("[bold]Model:[/bold]", result.model)
    metrics.add_row("[bold]Elapsed:[/bold]", f"{result.elapsed_seconds:.2f}s")
    metrics.add_row("[bold]Output tokens:[/bold]", _format_output_tokens(result.usage))
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


def _render_agent_result(result: AgentRunResult) -> None:
    if result.final_response:
        console.print(
            Panel(
                Markdown(result.final_response),
                title="[bold cyan]Poseidon Summary[/bold cyan]",
                border_style="cyan",
            )
        )

    metrics = Table(show_header=False, box=None, padding=(0, 2))
    metrics.add_row("[bold]Model:[/bold]", result.model)
    metrics.add_row("[bold]Model calls:[/bold]", f"{result.calls_made} / {result.max_calls}")
    metrics.add_row("[bold]Stop reason:[/bold]", result.stop_reason)
    if result.modified_files:
        metrics.add_row("[bold]Modified files:[/bold]", ", ".join(result.modified_files))
        verified_str = (
            "[bold green]VERIFIED[/bold green]"
            if result.edit_verified
            else "[bold red]UNVERIFIED[/bold red]"
        )
        metrics.add_row("[bold]Edit status:[/bold]", verified_str)
    metrics.add_row("[bold]Output tokens:[/bold]", _format_output_tokens(result.usage))
    metrics.add_row("[bold]Total tokens:[/bold]", _format_metric(result.usage.total_tokens))
    if result.estimated_tokens_saved > 0:
        metrics.add_row(
            "[bold]Context masked:[/bold]",
            f"~{result.estimated_tokens_saved} tokens saved (observation masking)",
        )
    metrics.add_row("[bold]Elapsed:[/bold]", f"{result.elapsed_seconds:.2f}s")
    metrics.add_row("[bold]Cost (USD):[/bold]", _format_cost(result.usage.cost_usd))
    trace_label = (
        "[green]sent[/green]"
        if result.traced_to_respan
        else "[dim]disabled (set RESPAN_API_KEY to enable)[/dim]"
    )
    metrics.add_row("[bold]Respan trace:[/bold]", trace_label)
    console.print(metrics)


def _run_agent_with_ui(
    *,
    client: OpenRouterClient,
    workspace: WorkspaceGuard,
    settings: Settings,
    model: str,
    prompt: str,
    max_calls: int,
    test_timeout: int,
    auto_approve: bool,
    tool_state: ToolSessionState | None = None,
    checkpoints: WorkspaceCheckpointManager | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
    tracer: PoseidonTracer | None = None,
    plan_mode: bool = False,
) -> AgentRunResult:
    """Execute the native tool-calling agent with live progress, diff preview, and Docker tests."""
    registry = ToolRegistry()
    hooks = HookManager()
    state = tool_state if tool_state is not None else ToolSessionState()
    test_runner = DockerTestRunner(
        image=settings.sandbox_image,
        timeout_seconds=test_timeout,
    )
    semantic_index = SemanticCodeIndex(
        workspace,
        FastEmbedProvider(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
        ),
    )

    register_core_tools(
        registry=registry,
        workspace=workspace,
        state=state,
        confirm_patch=lambda d: _preview_diff(d, auto_approve=auto_approve),
        test_runner=test_runner,
        on_test_result=_render_test_result,
        semantic_index=semantic_index,
        checkpoints=checkpoints,
        hooks=hooks,
        protect_tests=True,
        read_only=plan_mode,
    )

    plugin_report = load_workspace_plugins(workspace=workspace, registry=registry, hooks=hooks)
    for warn in plugin_report.warnings:
        console.print(f"[yellow]Plugin warning:[/yellow] {warn}")

    mcp_manager = MCPClientManager()
    mcp_report = mcp_manager.connect_and_register(workspace_root=workspace.root, registry=registry)
    for warn in mcp_report.warnings:
        console.print(f"[yellow]MCP warning:[/yellow] {warn}")

    memory = load_project_memory(workspace.root)
    sys_prompt = build_system_prompt(project_memory=memory.content, plan_mode=plan_mode)
    if conversation_history and conversation_history[0].get("role") == "system":
        conversation_history[0]["content"] = sys_prompt

    def _on_model_call(call_num: int, limit: int) -> None:
        tag = " (plan)" if plan_mode else ""
        console.print(f"[dim]→ Model call {call_num}/{limit}{tag} ({model})...[/dim]")

    def _on_tool_result(event: ToolEvent) -> None:
        icon = "[green]✓[/green]" if event.ok else "[red]✗[/red]"
        console.print(
            f"  {icon} [bold cyan]{event.tool_name}[/bold cyan] "
            f"[dim]({event.tool_call_id})[/dim] → {event.summary}"
        )

    active_tracer = tracer if tracer is not None else PoseidonTracer(settings)
    try:
        return run_agent_loop(
            client=client,
            registry=registry,
            model=model,
            user_prompt=prompt,
            max_calls=max_calls,
            tracer=active_tracer,
            conversation_history=conversation_history,
            on_model_call=_on_model_call,
            on_tool_result=_on_tool_result,
            workspace_root=workspace.root,
            tool_state=state,
            test_runner=test_runner,
            on_test_result=_render_test_result,
            system_prompt=sys_prompt,
        )
    finally:
        mcp_manager.close()


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


def _render_search_benchmark(report: SearchBenchmarkReport) -> None:
    """Display per-question grep vs. semantic search results and overall hit/latency summary."""
    console.print(
        f"[bold]Embedding model:[/bold] [green]{report.model_name}[/green]  |  "
        f"[bold]Indexed chunks:[/bold] {report.indexed_chunks}  |  "
        f"[bold]Index sync:[/bold] {report.indexing_ms:.1f}ms\n"
    )

    for res in report.results:
        grep_badge = "[bold green]HIT[/bold green]" if res.grep_hit else "[bold red]MISS[/bold red]"
        sem_badge = (
            "[bold green]HIT[/bold green]" if res.semantic_hit else "[bold red]MISS[/bold red]"
        )
        lines = [
            f"[bold]Question ({res.kind}):[/bold] {res.question}",
            f"[bold]Expected answer:[/bold] [yellow]{res.expected_target}[/yellow]",
            "",
            f"[bold cyan]Grep query:[/bold cyan] {res.grep_query!r}  →  "
            f"{grep_badge} ({res.grep_ms:.2f}ms)",
        ]
        if res.grep_top3:
            for idx, m in enumerate(res.grep_top3, start=1):
                lines.append(f"  {idx}. {m.qualified_id} ({m.location}) — {m.detail}")
        else:
            lines.append("  (no matches)")

        lines.append("")
        lines.append(
            f"[bold magenta]Semantic query:[/bold magenta] {res.semantic_query!r}  →  "
            f"{sem_badge} ({res.semantic_ms:.2f}ms)"
        )
        if res.semantic_top3:
            for idx, m in enumerate(res.semantic_top3, start=1):
                lines.append(f"  {idx}. {m.qualified_id} ({m.location}, {m.detail})")
        else:
            lines.append("  (no matches)")

        console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold]{res.id}[/bold]",
                border_style="bright_blue",
            )
        )

    summary = Table(
        title="Milestone 6 Summary — Grep (ripgrep) vs. Local Code Embeddings",
        border_style="cyan",
    )
    summary.add_column("Method", style="bold cyan")
    summary.add_column("Hits (Top-3)", justify="right")
    summary.add_column("Total Lookup Time (after indexing)", justify="right")
    summary.add_column("Avg Lookup Time", justify="right")
    n = max(1, report.total_questions)
    summary.add_row(
        "Grep (ripgrep)",
        f"{report.grep_hits} / {report.total_questions}",
        f"{report.grep_total_ms:.2f}ms",
        f"{report.grep_total_ms / n:.2f}ms",
    )
    summary.add_row(
        f"Embeddings ({report.model_name})",
        f"{report.semantic_hits} / {report.total_questions}",
        f"{report.semantic_total_ms:.2f}ms",
        f"{report.semantic_total_ms / n:.2f}ms",
    )
    console.print(summary)


def _render_model_comparison(rows: list[ModelComparisonRow]) -> None:
    """Display the Milestone 7 side-by-side model comparison table and run notes."""
    table = Table(
        title="Milestone 7 Comparison — Two Model Families + Hidden Acceptance Test",
        border_style="bright_blue",
    )
    table.add_column("Model ID", style="bold cyan", overflow="fold")
    table.add_column("Visible Tests", style="white", overflow="fold")
    table.add_column("Acceptance Test", style="white", overflow="fold")
    table.add_column("Calls", justify="right", no_wrap=True)
    table.add_column("Elapsed", justify="right", no_wrap=True)
    table.add_column("Tokens (Out / Total)", justify="right", no_wrap=True)
    table.add_column("Cost (USD)", justify="right", no_wrap=True)

    for row in rows:
        vis_style = "bold green" if row.visible_tests_passed else "bold red"
        acc_style = "bold green" if row.acceptance_passed else "bold red"
        tok_cell = f"{_format_metric(row.output_tokens)} / {_format_metric(row.total_tokens)}"
        table.add_row(
            row.model_id,
            f"[{vis_style}]{row.visible_tests_summary}[/{vis_style}]",
            f"[{acc_style}]{row.acceptance_summary}[/{acc_style}]",
            f"{row.calls_made} / {row.max_calls}",
            f"{row.elapsed_seconds:.2f}s",
            tok_cell,
            _format_cost(row.cost_usd),
        )
    console.print(table)

    for row in rows:
        files_str = ", ".join(row.modified_files) if row.modified_files else "(none)"
        detail = row.error or row.stop_reason
        console.print(
            f"[dim]• {row.model_id} | modified: {files_str} | note: {detail}[/dim]"
        )


@app.callback(invoke_without_command=True)
def root_callback(
    ctx: typer.Context,
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Project workspace folder (defaults to current)."),
    ] = Path("."),
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="OpenRouter model ID."),
    ] = None,
    plan: Annotated[
        bool,
        typer.Option("--plan", help="Start in read-only Plan Mode (write tools disabled)."),
    ] = False,
    continue_last: Annotated[
        bool,
        typer.Option("--continue", "-c", help="Resume the most recent saved session."),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Resume a specific saved session ID."),
    ] = None,
    max_calls: Annotated[
        int | None,
        typer.Option("--max-calls", help="Maximum model calls allowed per request."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Auto-approve diff patches without prompting."),
    ] = False,
) -> None:
    """Launch Poseidon-CLI interactively on the current project when run with no subcommand."""
    if ctx.invoked_subcommand is not None:
        return
    _start_interactive_session(
        workspace=workspace,
        model=model,
        mode="agent",
        max_calls=max_calls,
        test_timeout=None,
        yes=yes,
        plan=plan,
        continue_last=continue_last,
        resume=resume,
        prompt_for_model=False,
    )


@app.command()
def version() -> None:
    """Print the Poseidon-CLI version."""
    console.print(f"[bold cyan]{APP_NAME}[/bold cyan] v{__version__}")


def _render_tools_inventory(
    workspace: WorkspaceGuard,
    settings: Settings,
    *,
    plan_mode: bool = False,
) -> None:
    """Inspect and print all active built-in, plugin, and MCP tools + lifecycle hooks."""
    registry = ToolRegistry()
    hooks = HookManager()
    test_runner = DockerTestRunner(
        image=settings.sandbox_image,
        timeout_seconds=settings.default_test_timeout,
    )
    sem_index = SemanticCodeIndex(
        workspace,
        FastEmbedProvider(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
        ),
    )
    register_core_tools(
        registry=registry,
        workspace=workspace,
        state=ToolSessionState(),
        test_runner=test_runner,
        semantic_index=sem_index,
        hooks=hooks,
        read_only=plan_mode,
    )
    builtin_names = set(registry.names())

    plugin_report = load_workspace_plugins(workspace=workspace, registry=registry, hooks=hooks)
    plugin_names = set(plugin_report.registered_tools)

    with MCPClientManager() as mcp_manager:
        mcp_report = mcp_manager.connect_and_register(
            workspace_root=workspace.root,
            registry=registry,
        )
        mcp_names = set(mcp_report.registered_tools)

        table = Table(title=f"{APP_NAME} — Active Tools", border_style="cyan")
        table.add_column("Tool Name", style="bold cyan")
        table.add_column("Source", style="green")
        table.add_column("Description", style="white")
        for schema in registry.schemas():
            fn = schema["function"]
            name = fn["name"]
            if name in builtin_names:
                source = "built-in"
            elif name in plugin_names:
                source = "plugin"
            elif name in mcp_names:
                source = "mcp"
            else:
                source = "custom"
            table.add_row(name, source, fn["description"])
        console.print(table)

        hook_summary = hooks.summary()
        active_hooks = [f"{k}: {', '.join(v)}" for k, v in hook_summary.items() if v]
        if active_hooks:
            console.print(f"[cyan]Active hooks:[/cyan] {' | '.join(active_hooks)}")
        if mcp_report.connected_servers:
            console.print(
                f"[green]Connected MCP servers:[/green] {', '.join(mcp_report.connected_servers)}"
            )
        for warn in (*plugin_report.warnings, *mcp_report.warnings):
            console.print(f"[yellow]Warning:[/yellow] {warn}")


@app.command()
def tools(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Workspace to inspect for plugins and mcp.json."),
    ] = Path("."),
    plan: Annotated[
        bool,
        typer.Option("--plan", help="Show the read-only Plan Mode toolset."),
    ] = False,
) -> None:
    """List all active built-in, plugin, and MCP tools for a workspace."""
    settings = Settings()
    guard = WorkspaceGuard(workspace)
    _render_tools_inventory(guard, settings, plan_mode=plan)


@app.command()
def models() -> None:
    """List recommended free tool-capable OpenRouter models."""
    table = Table(title=f"{APP_NAME} — Recommended OpenRouter Models", border_style="cyan")
    table.add_column("Model ID", style="green")
    table.add_column("Notes", style="white")
    for model_id, desc in RECOMMENDED_MODELS:
        table.add_row(model_id, desc)
    console.print(table)


@app.command()
def test(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Workspace folder containing code and unit tests."),
    ] = Path("."),
    timeout: Annotated[
        int | None,
        typer.Option("--timeout", "-t", help="Test execution timeout in seconds."),
    ] = None,
    sandbox_image: Annotated[
        str | None,
        typer.Option("--sandbox-image", help="Docker image used for the isolated test sandbox."),
    ] = None,
) -> None:
    """Run workspace unit tests inside a disposable --network none Docker sandbox."""
    settings = Settings()
    effective_timeout = timeout if timeout is not None else settings.default_test_timeout
    effective_image = sandbox_image or settings.sandbox_image
    guard = WorkspaceGuard(workspace)

    runner = DockerTestRunner(
        image=effective_image,
        timeout_seconds=effective_timeout,
    )
    with console.status(
        f"[cyan]Running tests in isolated Docker sandbox (timeout {effective_timeout}s)...[/cyan]"
    ):
        result = runner.run(guard.root)
    _render_test_result(result)
    if not result.verified:
        raise typer.Exit(code=1)


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
            help="Optional relative filename (used in 'diff' or 'whole' mode).",
        ),
    ] = None,
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            help="Execution mode: 'agent' (native tools + Docker tests, default), 'diff', 'whole'.",
        ),
    ] = "agent",
    max_calls: Annotated[
        int | None,
        typer.Option(
            "--max-calls",
            help="Maximum model calls allowed per run (enforced in agent mode).",
        ),
    ] = None,
    test_timeout: Annotated[
        int | None,
        typer.Option(
            "--test-timeout",
            help="Isolated Docker test timeout in seconds.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Auto-approve diff patches without prompting."),
    ] = False,
    plan: Annotated[
        bool,
        typer.Option("--plan", help="Read-only Plan Mode (explore & plan without editing files)."),
    ] = False,
    no_code: Annotated[
        bool,
        typer.Option("--no-code", help="Suppress printing full source code in whole-file mode."),
    ] = False,
) -> None:
    """Execute a coding request in the sandboxed workspace."""
    settings = Settings()
    chosen_model = model or settings.default_model
    call_limit = max_calls if max_calls is not None else settings.default_max_calls
    effective_timeout = test_timeout if test_timeout is not None else settings.default_test_timeout

    if mode not in {"agent", "diff", "whole"}:
        console.print("[bold red]Error:[/bold red] --mode must be 'agent', 'diff', or 'whole'.")
        raise typer.Exit(code=1)

    try:
        guard = WorkspaceGuard(workspace)
    except WorkspaceSecurityError as exc:
        console.print(f"[bold red]Workspace error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    memory = load_project_memory(guard.root)
    render_banner(
        console,
        model=chosen_model,
        workspace=guard.root,
        plan_mode=plan,
        memory_sources=memory.sources,
    )

    with OpenRouterClient(settings) as client:
        try:
            if mode == "agent":
                chosen_model = _validate_or_reselect_model(
                    client, chosen_model, interactive=False
                )
                agent_res = _run_agent_with_ui(
                    client=client,
                    workspace=guard,
                    settings=settings,
                    model=chosen_model,
                    prompt=prompt,
                    max_calls=call_limit,
                    test_timeout=effective_timeout,
                    auto_approve=yes,
                    plan_mode=plan,
                )
                _render_agent_result(agent_res)
            elif mode == "diff":
                existing_files = guard.list_files()
                resolved_target = target_file or (
                    existing_files[0] if len(existing_files) == 1 else None
                )
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
        except ModelToolSupportError as exc:
            console.print(f"[bold red]Model error:[/bold red] {exc}")
            console.print(
                "[yellow]Tip: run `poseidon models` to pick a tool-capable model.[/yellow]"
            )
            raise typer.Exit(code=1) from exc
        except (OpenRouterError, FileExtractionError, WorkspaceSecurityError, ValueError) as exc:
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


@app.command(name="search-semantic")
def search_semantic(
    query: Annotated[str, typer.Argument(help="Plain-English query or symbol concept to find.")],
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Workspace folder to index and search."),
    ] = Path("."),
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of top matching code blocks to display."),
    ] = 3,
    exclude_tests: Annotated[
        bool,
        typer.Option("--exclude-tests", help="Exclude test_*.py files from results."),
    ] = False,
) -> None:
    """Index (or incrementally refresh) the workspace and search code by meaning."""
    settings = Settings()
    guard = WorkspaceGuard(workspace)
    provider = FastEmbedProvider(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
    )
    index = SemanticCodeIndex(guard, provider)

    with console.status(f"[cyan]Syncing & searching index ({settings.embedding_model})...[/cyan]"):
        result = index.search(query, top_k=top_k, auto_sync=True, exclude_tests=exclude_tests)

    stats = result.sync_stats
    sync_note = (
        "cache hit (0 files re-embedded)"
        if stats.cache_hit
        else (
            f"updated index (+{len(stats.added_files)} added, "
            f"~{len(stats.updated_files)} edited, -{len(stats.deleted_files)} deleted)"
        )
    )
    console.print(
        f"[dim]Index: {stats.total_chunks} chunk(s) across {stats.total_files} file(s) | "
        f"{sync_note} ({stats.elapsed_ms:.1f}ms) | Lookup: {result.lookup_ms:.2f}ms[/dim]"
    )

    table = Table(
        title=f"Semantic Search Results for {query!r}",
        border_style="cyan",
    )
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Location", style="yellow")
    table.add_column("Symbol", style="green")
    table.add_column("Kind", style="white")
    table.add_column("Score", justify="right")
    for idx, hit in enumerate(result.hits, start=1):
        table.add_row(
            str(idx),
            f"{hit.path}:{hit.start_line}-{hit.end_line}",
            hit.symbol,
            hit.kind,
            f"{hit.score:.3f}",
        )
    console.print(table)


@app.command(name="benchmark-search")
def benchmark_search(
    workspace: Annotated[
        Path,
        typer.Option("--workspace", "-w", help="Workspace folder containing the demo project."),
    ] = Path("."),
) -> None:
    """Run the Milestone 6 4-question benchmark comparing grep vs. local code embeddings."""
    settings = Settings()
    guard = WorkspaceGuard(workspace)
    render_banner(console, model=settings.embedding_model, workspace=guard.root)

    provider = FastEmbedProvider(
        model_name=settings.embedding_model,
        cache_dir=settings.embedding_cache_dir,
    )
    index = SemanticCodeIndex(guard, provider)

    with console.status(
        f"[cyan]Indexing workspace & running 4-question search benchmark "
        f"({settings.embedding_model})...[/cyan]"
    ):
        report = run_search_benchmark(workspace=guard, index=index)

    _render_search_benchmark(report)


@app.command(name="compare-models")
def compare_models(
    model_a: Annotated[
        str | None,
        typer.Option("--model-a", "-a", help="First tool-capable OpenRouter model ID."),
    ] = None,
    model_b: Annotated[
        str | None,
        typer.Option("--model-b", "-b", help="Second model ID from a different family."),
    ] = None,
    base_workspace: Annotated[
        Path,
        typer.Option(
            "--base-workspace",
            help="Clean base revision folder copied for each model run.",
        ),
    ] = Path("/app/benchmarks/m5_base"),
    acceptance_test: Annotated[
        Path,
        typer.Option(
            "--acceptance-test",
            help="Hidden acceptance test file (outside demo_workspace and search index).",
        ),
    ] = Path("/app/acceptance_tests/test_exclude_acceptance.py"),
    max_calls: Annotated[
        int,
        typer.Option("--max-calls", help="Identical model-call limit for both models."),
    ] = 12,
    test_timeout: Annotated[
        int,
        typer.Option("--test-timeout", help="Isolated Docker sandbox test timeout in seconds."),
    ] = 15,
    interactive: Annotated[
        bool,
        typer.Option("--interactive", "-i", help="Select both models interactively in the CLI."),
    ] = False,
) -> None:
    """Compare two tool-capable models from different families on the Milestone 5 request."""
    settings = Settings()

    with OpenRouterClient(settings) as client:
        first_model = (
            _select_model_interactive("nvidia/nemotron-3.5-super-vl:free")
            if (interactive or model_a is None)
            else model_a
        )
        first_model = _validate_or_reselect_model(client, first_model, interactive=True)

        second_model = (
            _select_model_interactive("google/gemma-4-31b-it:free")
            if (interactive or model_b is None)
            else model_b
        )
        second_model = _validate_or_reselect_model(client, second_model, interactive=True)

        render_banner(
            console,
            model=f"{first_model}  vs.  {second_model}",
            workspace=base_workspace,
        )
        console.print(f"[bold]Base revision:[/bold]    [yellow]{base_workspace}[/yellow]")
        console.print(f"[bold]Acceptance test:[/bold]  [yellow]{acceptance_test}[/yellow]")
        console.print(f"[bold]Call limit:[/bold]       {max_calls} calls per model\n")

        test_runner = DockerTestRunner(
            image=settings.sandbox_image,
            timeout_seconds=test_timeout,
        )
        emb_provider = FastEmbedProvider(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
        )
        tracer = PoseidonTracer(settings)

        def _on_start(mid: str) -> None:
            console.print(f"[bold cyan]▶ Running clean copy with {mid}...[/bold cyan]")

        try:
            rows = run_model_comparison(
                client=client,
                models=[first_model, second_model],
                base_workspace=base_workspace,
                acceptance_test_path=acceptance_test,
                test_runner=test_runner,
                embedding_provider=emb_provider,
                request=DEFAULT_M5_REQUEST,
                max_calls=max_calls,
                tracer=tracer,
                on_model_start=_on_start,
            )
        except FileNotFoundError as exc:
            console.print(f"[bold red]Comparison setup error:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc

    console.print()
    _render_model_comparison(rows)


def _render_challenge_table(rows: list[ChallengeRunRow]) -> None:
    table = Table(
        title="Optional Challenge — Baseline (v1) vs. +AST Preflight (v2)",
        border_style="bright_blue",
    )
    table.add_column("Task", style="bold white", overflow="fold")
    table.add_column("Harness Version", style="bold cyan", no_wrap=True)
    table.add_column("Visible Tests", overflow="fold")
    table.add_column("Acceptance Test", overflow="fold")
    table.add_column("Calls", justify="right", no_wrap=True)
    table.add_column("Elapsed", justify="right", no_wrap=True)
    table.add_column("Total Tokens", justify="right", no_wrap=True)

    for row in rows:
        vis_style = "bold green" if row.visible_passed else "bold red"
        acc_style = "bold green" if row.acceptance_passed else "bold red"
        table.add_row(
            row.task_name,
            row.harness_version,
            f"[{vis_style}]{row.visible_summary}[/{vis_style}]",
            f"[{acc_style}]{row.acceptance_summary}[/{acc_style}]",
            f"{row.calls_made} / {row.max_calls}",
            f"{row.elapsed_seconds:.2f}s",
            _format_metric(row.total_tokens),
        )
    console.print(table)


@app.command(name="benchmark-challenge")
def benchmark_challenge(
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="OpenRouter model ID for before/after evaluation."),
    ] = None,
    base_workspace: Annotated[
        Path,
        typer.Option("--base-workspace", help="Clean base revision folder."),
    ] = Path("/app/benchmarks/m5_base"),
    max_calls: Annotated[
        int,
        typer.Option("--max-calls", help="Identical call limit for every run."),
    ] = 10,
    test_timeout: Annotated[
        int,
        typer.Option("--test-timeout", help="Isolated Docker test timeout in seconds."),
    ] = 15,
) -> None:
    """Compare v1 (no preflight) vs. v2 (+AST preflight) on original and held-out tasks."""
    settings = Settings()
    chosen_model = model or settings.default_model

    with OpenRouterClient(settings) as client:
        chosen_model = _validate_or_reselect_model(client, chosen_model, interactive=False)
        render_banner(console, model=chosen_model, workspace=base_workspace)

        test_runner = DockerTestRunner(
            image=settings.sandbox_image,
            timeout_seconds=test_timeout,
        )
        tracer = PoseidonTracer(settings)

        def _on_step(ver: str, task_name: str) -> None:
            console.print(f"[bold cyan]▶ Running {ver} on {task_name}...[/bold cyan]")

        rows = run_optional_challenge_benchmark(
            client=client,
            model=chosen_model,
            base_workspace=base_workspace,
            exclude_acceptance_path=Path("/app/acceptance_tests/test_exclude_acceptance.py"),
            min_amount_acceptance_path=Path("/app/acceptance_tests/test_min_amount_heldout.py"),
            test_runner=test_runner,
            max_calls=max_calls,
            tracer=tracer,
            on_step=_on_step,
        )

    console.print()
    _render_challenge_table(rows)


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
            help="Execution mode: 'agent' (native tools + Docker tests, default), 'diff', 'whole'.",
        ),
    ] = "agent",
    max_calls: Annotated[
        int | None,
        typer.Option("--max-calls", help="Maximum model calls allowed per request."),
    ] = None,
    test_timeout: Annotated[
        int | None,
        typer.Option("--test-timeout", help="Isolated Docker test timeout in seconds."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Auto-approve diff patches without prompting."),
    ] = False,
    plan: Annotated[
        bool,
        typer.Option("--plan", help="Start in read-only Plan Mode (write tools disabled)."),
    ] = False,
    continue_last: Annotated[
        bool,
        typer.Option("--continue", "-c", help="Resume the most recent saved session."),
    ] = False,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Resume a specific saved session ID."),
    ] = None,
) -> None:
    """Start an interactive Poseidon-CLI session inside a workspace."""
    _start_interactive_session(
        workspace=workspace,
        model=model,
        mode=mode,
        max_calls=max_calls,
        test_timeout=test_timeout,
        yes=yes,
        plan=plan,
        continue_last=continue_last,
        resume=resume,
        prompt_for_model=(model is None),
    )


def _start_interactive_session(
    *,
    workspace: Path,
    model: str | None,
    mode: str,
    max_calls: int | None,
    test_timeout: int | None,
    yes: bool,
    plan: bool,
    continue_last: bool,
    resume: str | None,
    prompt_for_model: bool,
) -> None:
    """Shared interactive REPL used by both bare `poseidon` and `poseidon chat`."""
    settings = Settings()
    call_limit = max_calls if max_calls is not None else settings.default_max_calls
    effective_timeout = test_timeout if test_timeout is not None else settings.default_test_timeout
    auto_approve = yes
    plan_mode = plan

    try:
        guard = WorkspaceGuard(workspace)
    except WorkspaceSecurityError as exc:
        console.print(f"[bold red]Workspace error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    session_store = SessionStore(guard.root)
    session_id = generate_session_id()
    history: list[dict[str, Any]] = []
    chosen_model = model or (
        _select_model_interactive(settings.default_model)
        if prompt_for_model
        else settings.default_model
    )

    # Restore previous JSONL session if --resume or --continue was requested
    target_resume_id = resume or (session_store.latest_session_id() if continue_last else None)
    if target_resume_id:
        try:
            loaded = session_store.load(target_resume_id)
            session_id = loaded.summary.session_id
            history = loaded.messages
            if model is None and loaded.summary.model != "unknown":
                chosen_model = loaded.summary.model
            if not plan:
                plan_mode = loaded.summary.plan_mode
            console.print(
                f"[green]↺ Resumed session[/green] [cyan]{session_id}[/cyan] "
                f"({loaded.summary.message_count} messages)"
            )
        except FileNotFoundError as exc:
            console.print(f"[yellow]{exc}. Starting a fresh session instead.[/yellow]")

    tool_state = ToolSessionState()
    checkpoints = WorkspaceCheckpointManager(guard)
    tracer = PoseidonTracer(settings)

    with OpenRouterClient(settings) as client:
        if mode == "agent":
            chosen_model = _validate_or_reselect_model(client, chosen_model, interactive=True)

        memory = load_project_memory(guard.root)
        render_banner(
            console,
            model=chosen_model,
            workspace=guard.root,
            plan_mode=plan_mode,
            memory_sources=memory.sources,
            session_id=session_id,
        )
        console.print(
            f"[dim]Mode: {mode} | Max calls: {call_limit} | Test timeout: {effective_timeout}s | "
            f"Auto-edit: {auto_approve}[/dim]"
        )
        console.print(
            "[dim]Commands: /plan, /compact, /undo, /checkpoints, /save, /resume, /sessions, "
            "/memory, /init, /test, /model, /limit, /auto, /clear, /files, /exit[/dim]\n"
        )

        while True:
            prompt_label = (
                "[bold magenta]poseidon (plan)>[/bold magenta]"
                if plan_mode
                else "[bold cyan]poseidon>[/bold cyan]"
            )
            try:
                user_input = Prompt.ask(prompt_label).strip()
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
            if user_input == "/tools":
                _render_tools_inventory(guard, settings, plan_mode=plan_mode)
                continue
            if user_input == "/plan":
                plan_mode = not plan_mode
                status_text = (
                    "[bold magenta]ON (read-only; write tools disabled)[/bold magenta]"
                    if plan_mode
                    else "[bold green]OFF (edit tools enabled)[/bold green]"
                )
                console.print(f"Plan Mode: {status_text}")
                continue
            if user_input == "/memory":
                memory = load_project_memory(guard.root)
                if memory.sources:
                    console.print(
                        f"[cyan]Loaded project memory:[/cyan] {', '.join(memory.sources)} "
                        f"({len(memory.content)} chars)"
                    )
                else:
                    console.print(
                        "[dim]No POSEIDON.md or AGENTS.md found. Run `/init` to create one.[/dim]"
                    )
                continue
            if user_input == "/init":
                target = guard.root / "POSEIDON.md"
                if target.exists():
                    console.print("[yellow]POSEIDON.md already exists in this workspace.[/yellow]")
                else:
                    guard.write_text("POSEIDON.md", starter_poseidon_md(guard.root.name))
                    console.print("[green]✓ Created POSEIDON.md in workspace.[/green]")
                continue
            if user_input == "/sessions":
                saved = session_store.list_sessions()
                if not saved:
                    console.print("[dim]No saved sessions in this workspace yet.[/dim]")
                else:
                    tbl = Table(title="Saved Sessions", border_style="cyan")
                    tbl.add_column("Session ID", style="cyan")
                    tbl.add_column("Messages", justify="right")
                    tbl.add_column("Model", style="green")
                    tbl.add_column("Updated", style="dim")
                    tbl.add_column("Title", style="white")
                    for s in saved[:15]:
                        tbl.add_row(
                            s.session_id,
                            str(s.message_count),
                            s.model,
                            s.updated_at,
                            s.title,
                        )
                    console.print(tbl)
                continue
            if user_input.startswith("/save"):
                parts = user_input.split(maxsplit=1)
                custom_id = parts[1].strip() if len(parts) == 2 else session_id
                try:
                    saved_meta = session_store.save(
                        session_id=custom_id,
                        model=chosen_model,
                        messages=history,
                        plan_mode=plan_mode,
                    )
                    session_id = saved_meta.session_id
                    console.print(
                        f"[green]✓ Saved session[/green] [cyan]{saved_meta.session_id}[/cyan] "
                        f"({saved_meta.message_count} messages)"
                    )
                except ValueError as exc:
                    console.print(f"[red]Save failed:[/red] {exc}")
                continue
            if user_input.startswith("/resume"):
                parts = user_input.split(maxsplit=1)
                target_id = (
                    parts[1].strip() if len(parts) == 2 else session_store.latest_session_id()
                )
                if not target_id:
                    console.print("[yellow]No saved sessions to resume.[/yellow]")
                    continue
                try:
                    loaded = session_store.load(target_id)
                    session_id = loaded.summary.session_id
                    history = loaded.messages
                    plan_mode = loaded.summary.plan_mode
                    console.print(
                        f"[green]↺ Resumed session[/green] [cyan]{session_id}[/cyan] "
                        f"({loaded.summary.message_count} messages)"
                    )
                except (FileNotFoundError, ValueError) as exc:
                    console.print(f"[red]Resume failed:[/red] {exc}")
                continue
            if user_input == "/clear":
                history.clear()
                session_id = generate_session_id()
                console.print(
                    f"[green]✓ Cleared history. New session:[/green] [cyan]{session_id}[/cyan]"
                )
                continue
            if user_input == "/compact":
                _, stats = compact_messages(
                    history,
                    keep_recent_tool_outputs=2,
                    min_chars_to_mask=150,
                    in_place=True,
                )
                console.print(
                    f"[green]✓ Context compacted:[/green] masked {stats.masked_count} old tool "
                    f"output(s), saved {stats.chars_saved} chars (~{stats.estimated_tokens_saved} "
                    f"tokens; {stats.chars_before} → {stats.chars_after} chars)."
                )
                continue
            if user_input == "/checkpoints":
                cps = checkpoints.list_checkpoints()
                if not cps:
                    console.print("[dim]No edit checkpoints captured yet.[/dim]")
                else:
                    for cp in cps:
                        console.print(f"  [cyan]#{cp.id}[/cyan] ({cp.created_at}) — {cp.label}")
                continue
            if user_input == "/undo":
                cp, restored = checkpoints.undo()
                if cp is None:
                    console.print("[yellow]Nothing to undo (checkpoint stack is empty).[/yellow]")
                else:
                    tool_state.read_hashes.clear()
                    tool_state.dirty_since_last_test = True
                    console.print(
                        f"[bold green]↺ Undid checkpoint #{cp.id}[/bold green] "
                        f"({cp.label}): {', '.join(restored)}"
                    )
                continue
            if user_input == "/test":
                runner = DockerTestRunner(
                    image=settings.sandbox_image,
                    timeout_seconds=effective_timeout,
                )
                res = runner.run(guard.root)
                _render_test_result(res)
                continue
            if user_input == "/model":
                selected = _select_model_interactive(chosen_model)
                chosen_model = (
                    _validate_or_reselect_model(client, selected, interactive=True)
                    if mode == "agent"
                    else selected
                )
                console.print(f"[green]Active model switched to:[/green] {chosen_model}")
                continue
            if user_input.startswith("/limit"):
                parts = user_input.split()
                if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= 1:
                    call_limit = int(parts[1])
                    console.print(f"[green]Model-call limit set to:[/green] {call_limit}")
                else:
                    console.print("[yellow]Usage: /limit <positive_integer>[/yellow]")
                continue
            if user_input.startswith("/timeout"):
                parts = user_input.split()
                if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= 1:
                    effective_timeout = int(parts[1])
                    console.print(f"[green]Test timeout set to:[/green] {effective_timeout}s")
                else:
                    console.print("[yellow]Usage: /timeout <seconds>[/yellow]")
                continue
            if user_input.startswith("/auto"):
                parts = user_input.split()
                if len(parts) == 2 and parts[1] in {"on", "off"}:
                    auto_approve = parts[1] == "on"
                    console.print(f"[green]Auto-approve edits:[/green] {auto_approve}")
                else:
                    console.print("[yellow]Usage: /auto <on|off>[/yellow]")
                continue
            if user_input.startswith("/mode"):
                parts = user_input.split()
                if len(parts) == 2 and parts[1] in {"agent", "diff", "whole"}:
                    mode = parts[1]
                    console.print(f"[green]Mode switched to:[/green] {mode}")
                else:
                    console.print("[yellow]Usage: /mode <agent|diff|whole>[/yellow]")
                continue

            try:
                if mode == "agent":
                    agent_res = _run_agent_with_ui(
                        client=client,
                        workspace=guard,
                        settings=settings,
                        model=chosen_model,
                        prompt=user_input,
                        max_calls=call_limit,
                        test_timeout=effective_timeout,
                        auto_approve=auto_approve,
                        tool_state=tool_state,
                        checkpoints=checkpoints,
                        conversation_history=history,
                        tracer=tracer,
                        plan_mode=plan_mode,
                    )
                    _render_agent_result(agent_res)
                    # Auto-save session cleanly after every completed turn
                    session_store.save(
                        session_id=session_id,
                        model=chosen_model,
                        messages=history,
                        plan_mode=plan_mode,
                    )
                elif mode == "diff":
                    existing_files = guard.list_files()
                    if not existing_files:
                        console.print(
                            "[yellow]Workspace is empty; switch to /mode whole first.[/yellow]"
                        )
                        continue
                    diff_res = _run_diff_with_spinner(
                        client=client,
                        workspace=guard,
                        model=chosen_model,
                        prompt=user_input,
                        target_file=existing_files[0],
                        auto_approve=auto_approve,
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
            except ModelToolSupportError as exc:
                console.print(f"[bold red]Model error:[/bold red] {exc}")
                chosen_model = _select_model_interactive(chosen_model)
            except (
                OpenRouterError,
                FileExtractionError,
                WorkspaceSecurityError,
                ValueError,
            ) as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")


def main() -> None:
    """Console script entrypoint."""
    app()


if __name__ == "__main__":
    main()
