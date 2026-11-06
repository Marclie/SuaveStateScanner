# AGENTS.md

## Project overview

This repository contains SuaveStateScanner, a Python tool for labeling and preserving continuity of excited-state curves along a reaction coordinate. The core algorithm lives in `suave_state_scanner.py`, while the interactive browser UI is implemented under `suave_gui/`.

The authoritative user-facing docs are in [README.md](README.md); prefer those semantics when changing CLI arguments, config keys, or GUI behavior.

## Repository map

- `suave_state_scanner.py`: main CLI/backend implementation, finite-difference ordering logic, and configuration parsing.
- `suave_gui/`: Dash-based GUI, dataset parser, plotting, session state, and sidebar controls.
- `suave_gui_runner.py`: GUI entry point.
- `suave_metric.py` and `nstencil.py`: metric and stencil helpers used by the scanner.
- `tests/`: unit and integration tests for dataset parsing, numerical behavior, GUI sync, and save/export paths.
- `suave.json`: default configuration example.
- `input.csv`, `output.csv`, `checkpoint.csv`: sample or generated workflow artifacts.

## Working conventions

- Keep the CLI and GUI behavior aligned when a change affects data loading, config keys, matrix shape assumptions, or output ordering.
- Preserve compatibility with the CSV input format and JSON config fields documented in [README.md](README.md).
- Prefer surgical edits in the relevant module instead of broad refactors.
- Follow the existing NumPy/Numba patterns used in the scanner; avoid introducing unnecessary dependencies.
- When numerical behavior changes, add or update tests in `tests/` alongside the fix.

## Commands

Run from the repository root:

- Install dependencies: `python -m pip install -r requirements.txt`
- Run the full suite: `pytest` or `python -m pytest`
- Run a focused file: `pytest tests/unit/test_dataset.py`
- Launch the GUI: `python suave_gui_runner.py`
- Run the CLI: `python suave_state_scanner.py [-c CONFIG_PATH] in_file out_file num_states`

## Testing expectations

This project uses pytest with the repository root as the import path (`pytest.ini`). The test suite emphasizes:

- numerical equivalence and scanner correctness
- GUI/backend synchronization
- save/export and checkpoint behavior
- parser and dataset validation

If a fix touches state ordering, interpolation, or UI-driven sweep logic, validate with the closest relevant tests before calling the work complete.

## Common pitfalls

- The scanner expects a matrix with reaction coordinate + state energies/properties; changing column layout or indexing assumptions can silently break ordering.
- Config values are passed through `SuaveStateScanner` and the GUI shares the same backend object, so updating config names without updating both code paths will produce mismatches.
- Generated CSV files such as `checkpoint.csv` or `output.csv` are workflow artifacts; do not treat them as source-of-truth code.

## Helpful references

- [README.md](README.md)
- [suave.json](suave.json)
- [pytest.ini](pytest.ini)
