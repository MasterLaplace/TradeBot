#!/usr/bin/env python3
"""
TradeBot - Unified Trading Bot Entry Point.

Usage:
    python tradebot.py --help
    python tradebot.py backtest --data data/crypto.csv --strategy safe_profit
    python tradebot.py paper --duration 3600
    python tradebot.py fetch --days 30 --output data/crypto.csv

Or use as module:
    python -m src.cli --help
"""

import sys
from pathlib import Path

# Add src to path for clean imports
sys.path.insert(0, str(Path(__file__).parent))

from src.cli import main

if __name__ == '__main__':
    sys.exit(main())
