"""
CLI Module - Entry Point.

Provides unified command-line interface for all trading bot operations.
"""

from .main import main, create_parser
from .commands import (
    handle_backtest,
    handle_compare,
    handle_paper,
    handle_fetch,
    handle_report,
    handle_list,
    handle_test,
)

__all__ = [
    "main",
    "create_parser",
    "handle_backtest",
    "handle_compare",
    "handle_paper",
    "handle_fetch",
    "handle_report",
    "handle_list",
    "handle_test",
]
