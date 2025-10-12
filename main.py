"""Command line entry point for running the MiDA optimisation pipeline."""

from argparse import ArgumentParser
from typing import Iterable, Optional

from read_log import ReadLog
from MiDA import MiDA


def _build_parser() -> ArgumentParser:
    """Create the argument parser used by the CLI."""

    parser = ArgumentParser(
        description="Run the MiDA optimisation pipeline for the provided event log."
    )
    parser.add_argument(
        "eventlog",
        help=(
            "Name of the event log (used to locate the pre-processed fold data), "
            "for example 'bpi12w_complete'."
        ),
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Run the CLI with the provided arguments.

    Parameters
    ----------
    argv:
        Iterable of command-line arguments. When ``None`` (the default), ``sys.argv``
        is used which matches the behaviour of a standard command-line entry point.
    """

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    ReadLog(args.eventlog).readView()
    MiDA(args.eventlog).optimize()


if __name__ == "__main__":
    main()

