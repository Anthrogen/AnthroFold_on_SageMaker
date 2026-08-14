#!/usr/bin/env python3
"""Safely delete the AnthroFold MSA endpoint, config, and model.

Equivalent to ``python src/msa_endpoint.py delete ...``. It prints the exact
resources and asks for confirmation; pass ``--yes`` only for non-interactive use.
"""

from msa_endpoint import main


if __name__ == "__main__":
    raise SystemExit(main(default_command="delete"))
