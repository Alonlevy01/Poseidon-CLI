"""Dynamic loader for workspace-local Python plugins under `.poseidon/plugins/*.py`."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

from poseidon.plugins.hooks import HookManager, PostEditHook, PostTestHook, PreEditHook
from poseidon.tools.registry import ToolDefinition, ToolRegistry
from poseidon.workspace import WorkspaceGuard

PLUGINS_SUBDIR = ".poseidon/plugins"


@dataclass
class PluginLoadReport:
    """Summary of discovered plugins, registered tools, hooks, and any load warnings."""

    loaded_plugins: list[str] = field(default_factory=list)
    registered_tools: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PluginAPI:
    """Controlled registration surface exposed to each plugin's `register(api)` function."""

    def __init__(
        self,
        *,
        plugin_name: str,
        workspace: WorkspaceGuard,
        registry: ToolRegistry,
        hooks: HookManager,
        report: PluginLoadReport,
    ) -> None:
        self.plugin_name = plugin_name
        self.workspace = workspace
        self._registry = registry
        self._hooks = hooks
        self._report = report

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register a custom ToolDefinition into Poseidon's ToolRegistry."""
        self._registry.register(tool)
        self._report.registered_tools.append(tool.name)

    def on_pre_edit(self, fn: PreEditHook, *, name: str | None = None) -> None:
        self._hooks.add_pre_edit(name or self.plugin_name, fn)

    def on_post_edit(self, fn: PostEditHook, *, name: str | None = None) -> None:
        self._hooks.add_post_edit(name or self.plugin_name, fn)

    def on_post_test(self, fn: PostTestHook, *, name: str | None = None) -> None:
        self._hooks.add_post_test(name or self.plugin_name, fn)


def load_workspace_plugins(
    *,
    workspace: WorkspaceGuard,
    registry: ToolRegistry,
    hooks: HookManager,
) -> PluginLoadReport:
    """Discover and load Python plugins from `<workspace>/.poseidon/plugins/*.py`."""
    report = PluginLoadReport()
    plugins_dir = workspace.root / PLUGINS_SUBDIR
    if not plugins_dir.is_dir():
        return report

    for path in sorted(plugins_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        plugin_name = path.stem
        try:
            spec = importlib.util.spec_from_file_location(
                f"poseidon_plugin_{plugin_name}",
                path,
            )
            if spec is None or spec.loader is None:
                report.warnings.append(f"Skipped '{path.name}': could not create module spec.")
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            register_fn = getattr(module, "register", None)
            if not callable(register_fn):
                report.warnings.append(f"Skipped '{path.name}': missing callable `register(api)`.")
                continue

            api = PluginAPI(
                plugin_name=plugin_name,
                workspace=workspace,
                registry=registry,
                hooks=hooks,
                report=report,
            )
            register_fn(api)
            report.loaded_plugins.append(plugin_name)
        except Exception as exc:
            report.warnings.append(f"Plugin '{plugin_name}' failed to load: {exc}")

    return report
