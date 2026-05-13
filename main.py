#!/usr/bin/env python3
"""Convenience script: `python main.py` equivalent to `python -m basin`."""

import sys
from basin.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
