"""Disposable Docker sandbox runner for untrusted generated code and unit tests.

Security & isolation guarantees:
- Fresh ephemeral container built from official python:3.12-slim (`poseidon-sandbox:dev`)
- `network_mode="none"` (zero network access)
- `environment={}` (zero secrets passed from harness)
- `cap_drop=["ALL"]`, `security_opt=["no-new-privileges:true"]`, non-root `sandbox` user
- Resource caps: 512MB RAM, 1 CPU, 64 PIDs, hard wall-clock timeout
- Workspace files copied in via tar archive (never bind-mounted), container always removed
"""

from __future__ import annotations

import contextlib
import io
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

EXCLUDED_SANDBOX_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".poseidon",
    ".poseidon_cache",
    "node_modules",
}


class TestFailure(BaseModel):
    """Name and error detail for a single failing or errored unit test."""

    __test__ = False

    name: str
    message: str


class TestRunResult(BaseModel):
    """Structured outcome of an isolated unit test execution."""

    __test__ = False

    command: str
    timeout_seconds: int
    exit_code: int | None = None
    timed_out: bool = False
    runner_error: str | None = None
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    total: int = 0
    failures: list[TestFailure] = Field(default_factory=list)
    output: str = ""
    elapsed_seconds: float = 0.0
    verified: bool = False
    unverified_reason: str | None = None

    def format_for_model(self, max_output_chars: int = 4000) -> str:
        """Format test status, counts, failures, and trimmed output for the LLM."""
        lines = [
            f"Command: {self.command}",
            f"Timeout: {self.timeout_seconds}s | Exit status: {self.exit_code}",
            f"Status: {'VERIFIED (all tests passed)' if self.verified else 'UNVERIFIED'}",
            f"Counts: passed={self.passed}, failed={self.failed}, "
            f"errors={self.errors}, skipped={self.skipped}, total={self.total}",
        ]
        if self.unverified_reason:
            lines.append(f"Reason: {self.unverified_reason}")
        if self.failures:
            lines.append("Failing tests:")
            for fail in self.failures:
                lines.append(f"  - {fail.name}: {fail.message}")
        if self.output.strip():
            trimmed = self.output.strip()
            if len(trimmed) > max_output_chars:
                trimmed = trimmed[:max_output_chars] + "\n... [output truncated]"
            lines.append("\nTest output:\n" + trimmed)
        return "\n".join(lines)


class TestRunner(Protocol):
    """Protocol for any isolated unit-test runner (Docker or mocked in harness tests)."""

    @property
    def timeout_seconds(self) -> int: ...

    def run(
        self,
        workspace_root: Path,
        *,
        test_args: list[str] | None = None,
        extra_files: dict[str, str] | None = None,
    ) -> TestRunResult: ...


def build_workspace_tar(
    workspace_root: Path,
    *,
    extra_files: dict[str, str] | None = None,
) -> bytes:
    """Pack clean workspace files (excluding secrets/hidden files/caches) into a tar archive."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path in sorted(workspace_root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(workspace_root)
            if any(part.startswith(".") or part in EXCLUDED_SANDBOX_DIRS for part in rel.parts):
                continue
            data = path.read_bytes()
            info = tarfile.TarInfo(name=rel.as_posix())
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))

        if extra_files:
            for rel_name, content in extra_files.items():
                data = content.encode("utf-8")
                info = tarfile.TarInfo(name=rel_name)
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))

    return buf.getvalue()


def parse_junit_xml(xml_text: str) -> tuple[int, int, int, int, list[TestFailure]]:
    """Parse pytest JUnit XML into (passed, failed, errors, skipped, failures)."""
    root = ET.fromstring(xml_text)
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]

    total = 0
    failed = 0
    errors = 0
    skipped = 0
    failures: list[TestFailure] = []

    for suite in suites:
        total += int(suite.attrib.get("tests", 0))
        failed += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        skipped += int(suite.attrib.get("skipped", 0))

        for case in suite.findall("testcase"):
            cls = case.attrib.get("classname", "")
            name = case.attrib.get("name", "unknown")
            full_name = f"{cls}::{name}" if cls else name

            for tag in ("failure", "error"):
                node = case.find(tag)
                if node is not None:
                    msg = (node.attrib.get("message") or "").strip()
                    body = (node.text or "").strip()
                    detail = msg or (body.splitlines()[-1] if body else f"Test {tag}")
                    failures.append(TestFailure(name=full_name, message=detail))

    passed = max(0, total - failed - errors - skipped)
    return passed, failed, errors, skipped, failures


def evaluate_test_outcome(
    *,
    command: str,
    timeout_seconds: int,
    exit_code: int | None,
    timed_out: bool,
    runner_error: str | None,
    xml_text: str | None,
    output: str,
    elapsed_seconds: float,
) -> TestRunResult:
    """Build a verified/unverified TestRunResult from raw execution artifacts."""
    if runner_error:
        return TestRunResult(
            command=command,
            timeout_seconds=timeout_seconds,
            exit_code=exit_code,
            runner_error=runner_error,
            output=output,
            elapsed_seconds=elapsed_seconds,
            verified=False,
            unverified_reason=f"Test command could not run: {runner_error}",
        )

    if timed_out:
        return TestRunResult(
            command=command,
            timeout_seconds=timeout_seconds,
            exit_code=None,
            timed_out=True,
            output=output,
            elapsed_seconds=elapsed_seconds,
            verified=False,
            unverified_reason=f"Test run timed out after {timeout_seconds}s.",
        )

    passed = failed = errors = skipped = 0
    failures: list[TestFailure] = []
    if xml_text:
        with contextlib.suppress(ET.ParseError):
            passed, failed, errors, skipped, failures = parse_junit_xml(xml_text)

    total = passed + failed + errors + skipped

    if exit_code == 5 or (exit_code == 0 and total == 0):
        return TestRunResult(
            command=command,
            timeout_seconds=timeout_seconds,
            exit_code=exit_code,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total=total,
            failures=failures,
            output=output,
            elapsed_seconds=elapsed_seconds,
            verified=False,
            unverified_reason="No tests ran (0 tests collected).",
        )

    if exit_code == 0 and total > 0 and failed == 0 and errors == 0:
        return TestRunResult(
            command=command,
            timeout_seconds=timeout_seconds,
            exit_code=0,
            passed=passed,
            failed=0,
            errors=0,
            skipped=skipped,
            total=total,
            failures=[],
            output=output,
            elapsed_seconds=elapsed_seconds,
            verified=True,
            unverified_reason=None,
        )

    if failed > 0 or errors > 0:
        reason = f"{failed} test(s) failed, {errors} error(s) (exit status {exit_code})."
    elif total == 0:
        reason = f"No tests completed cleanly (exit status {exit_code})."
    else:
        reason = f"Test command exited with non-zero status {exit_code}."

    return TestRunResult(
        command=command,
        timeout_seconds=timeout_seconds,
        exit_code=exit_code,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        total=total,
        failures=failures,
        output=output,
        elapsed_seconds=elapsed_seconds,
        verified=False,
        unverified_reason=reason,
    )


class DockerTestRunner:
    """Runs pytest inside an ephemeral, network-isolated, secret-free Docker container."""

    def __init__(
        self,
        *,
        image: str = "poseidon-sandbox:dev",
        timeout_seconds: int = 15,
        mem_limit: str = "512m",
        nano_cpus: int = 1_000_000_000,
        pids_limit: int = 64,
    ) -> None:
        self._image = image
        self._timeout_seconds = timeout_seconds
        self._mem_limit = mem_limit
        self._nano_cpus = nano_cpus
        self._pids_limit = pids_limit

    @property
    def timeout_seconds(self) -> int:
        return self._timeout_seconds

    def run(
        self,
        workspace_root: Path,
        *,
        test_args: list[str] | None = None,
        extra_files: dict[str, str] | None = None,
    ) -> TestRunResult:
        """Execute unit tests in a disposable container with --network none and zero secrets."""
        import docker
        from docker.errors import DockerException, ImageNotFound, NotFound

        args = list(test_args) if test_args else []
        display_cmd = " ".join(["pytest", "-v", "--tb=short", *args])
        container_cmd = ["pytest", "-v", "--tb=short", "--junitxml=/tmp/junit.xml", *args]
        started = time.perf_counter()

        try:
            client = docker.from_env()
        except DockerException as exc:
            return evaluate_test_outcome(
                command=display_cmd,
                timeout_seconds=self._timeout_seconds,
                exit_code=None,
                timed_out=False,
                runner_error=f"Cannot connect to Docker daemon: {exc}",
                xml_text=None,
                output="",
                elapsed_seconds=time.perf_counter() - started,
            )

        container = None
        try:
            tar_bytes = build_workspace_tar(workspace_root, extra_files=extra_files)
            container = client.containers.create(
                image=self._image,
                command=container_cmd,
                working_dir="/sandbox",
                user="sandbox",
                environment={},
                network_mode="none",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                mem_limit=self._mem_limit,
                nano_cpus=self._nano_cpus,
                pids_limit=self._pids_limit,
            )
            container.put_archive("/sandbox", tar_bytes)
            container.start()

            timed_out = False
            exit_code: int | None = None
            try:
                wait_result = container.wait(timeout=self._timeout_seconds)
                exit_code = int(wait_result.get("StatusCode", 1))
            except Exception as exc:
                # requests/urllib3 ReadTimeout or ConnectionError on container.wait timeout
                if "timed out" in str(exc).lower() or "timeout" in type(exc).__name__.lower():
                    timed_out = True
                    with contextlib.suppress(DockerException):
                        container.kill()
                else:
                    raise

            raw_logs = container.logs(stdout=True, stderr=True)
            output = raw_logs.decode("utf-8", errors="replace")

            xml_text: str | None = None
            if not timed_out:
                try:
                    stream, _ = container.get_archive("/tmp/junit.xml")
                    xml_tar = io.BytesIO(b"".join(stream))
                    with tarfile.open(fileobj=xml_tar, mode="r") as tf:
                        member = tf.getmembers()[0]
                        extracted = tf.extractfile(member)
                        if extracted is not None:
                            xml_text = extracted.read().decode("utf-8", errors="replace")
                except (NotFound, tarfile.TarError, IndexError):
                    xml_text = None

            return evaluate_test_outcome(
                command=display_cmd,
                timeout_seconds=self._timeout_seconds,
                exit_code=exit_code,
                timed_out=timed_out,
                runner_error=None,
                xml_text=xml_text,
                output=output,
                elapsed_seconds=time.perf_counter() - started,
            )

        except ImageNotFound:
            return evaluate_test_outcome(
                command=display_cmd,
                timeout_seconds=self._timeout_seconds,
                exit_code=None,
                timed_out=False,
                runner_error=(
                    f"Sandbox image '{self._image}' not found. "
                    "Build it with: docker compose --profile build-only build poseidon-sandbox"
                ),
                xml_text=None,
                output="",
                elapsed_seconds=time.perf_counter() - started,
            )
        except DockerException as exc:
            return evaluate_test_outcome(
                command=display_cmd,
                timeout_seconds=self._timeout_seconds,
                exit_code=None,
                timed_out=False,
                runner_error=f"Docker sandbox error: {exc}",
                xml_text=None,
                output="",
                elapsed_seconds=time.perf_counter() - started,
            )
        finally:
            if container is not None:
                with contextlib.suppress(DockerException):
                    container.remove(force=True)
            client.close()
