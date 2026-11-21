# Poseidon-CLI

A modular Python CLI coding agent harness built for clean software engineering workflows.

## Quick Start (Docker)

1. Copy `.env.example` to `.env` and set your `OPENROUTER_API_KEY`:
   ```bash
   cp .env.example .env
   ```
2. Build the container image (based on official `python:3.12-slim`):
   ```bash
   docker compose build
   ```
3. Inspect CLI help:
   ```bash
   docker compose run --rm poseidon --help
   ```
4. Start an interactive session inside `demo_workspace/`:
   ```bash
   docker compose run --rm poseidon chat --workspace /workspace
   ```
5. Run a one-shot generation request:
   ```bash
   docker compose run --rm poseidon run --workspace /workspace --model nvidia/nemotron-3.5-super-vl:free "Create an expense ledger CLI..."
   ```
6. Run the harness unit tests & linter inside the container:
   ```bash
   docker compose run --rm --entrypoint pytest poseidon /app/tests
   docker compose run --rm --entrypoint ruff poseidon check /app/src /app/tests
   ```
