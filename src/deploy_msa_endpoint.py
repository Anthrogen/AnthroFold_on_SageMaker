#!/usr/bin/env python3
"""Friendly entry point for deploying the AnthroFold MSA endpoint.

Equivalent to ``python src/msa_endpoint.py deploy ...``. Use the all-in-one
``msa_endpoint.py`` command when overriding lifecycle state/sidecar locations.
"""

from msa_endpoint import main


if __name__ == "__main__":
    raise SystemExit(main(default_command="deploy"))
