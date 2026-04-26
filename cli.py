#!/usr/bin/env python3
"""CLI entry point - delegates to app.cli.main."""

import sys

from app.cli import main

if __name__ == "__main__":
    sys.exit(main())
