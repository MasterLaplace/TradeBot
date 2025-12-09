"""
CLI Module - Entry Point.

Provides unified command-line interface for all trading bot operations.
"""

from .main import main, create_parser

__all__ = [
    "main",
    "create_parser",
]
