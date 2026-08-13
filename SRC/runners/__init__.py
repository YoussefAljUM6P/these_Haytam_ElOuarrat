"""SERVIS experiment runners.

Each module exposes:
    add_arguments(parser) -> None   # registers argparse flags on a subparser
    run(args) -> int | None         # executes the experiment using parsed args

Invoked from `SRC/cli.py`; not meant to be run directly. Functionality used
to live in top-level `main_*.py` / `run_*.py` scripts.
"""
