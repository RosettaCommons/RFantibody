#!/usr/bin/env python3

"""
Root conftest file for all pytest tests.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--keep-outputs", 
        action="store_true", 
        default=False,
        help="Keep test outputs in standard directory instead of using temporary directory"
    )