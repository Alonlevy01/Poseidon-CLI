# workspace — Poseidon Project Memory

## Architecture & Conventions
- Keep functions small, typed, and well-documented in English.
- Preserve clean module boundaries (separate parsing, logic, formatting, and CLI).

## Testing & Verification
- Run unit tests (`pytest`) inside the isolated Docker sandbox after edits.
- Never modify existing unit tests to mask an implementation bug.
