"""Command-line interface."""

from __future__ import annotations

import click


@click.command()
@click.version_option()
def main() -> None:
    """Use Super-Droplet Model and EUREC4A data to simulate rain evaporation."""


if __name__ == "__main__":
    main(prog_name="sdm_eurec4a")  # pragma: no cover
