#!/usr/bin/env python3
"""
TradeBot — launcher for the interactive console.

Equivalent to the installed `tradebot` command. No arguments: it opens the
interactive console (REPL) where you type commands. See SPEC.md §9.

Usage:
    python tradebot.py        # or, once installed:  tradebot
"""

import sys
from pathlib import Path

# Allow running straight from a checkout without installing.
sys.path.insert(0, str(Path(__file__).parent))

from src.console import main

if __name__ == "__main__":
    main()
