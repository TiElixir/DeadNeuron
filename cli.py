"""Command line entry point: `python cli.py ...` is the same as `python -m neuronlab.cli ...`."""
from neuronlab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
