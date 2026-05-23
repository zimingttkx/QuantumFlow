"""Integration tests for quantumflow failover module.

This module contains integration tests that verify component interactions
in a real or near-real environment.
"""

import pytest

# Configure asyncio mode for all integration tests
pytest_plugins = ["tests.integration.failover.conftest"]
