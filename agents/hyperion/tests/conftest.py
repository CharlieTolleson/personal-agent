"""
Suite-wide pytest setup for Hyperion.

Sets ``HYPERION_API_TOKEN`` for every test so the bearer-token middleware
(``hyperion.server.auth``) has a configured token; tests that check the
"no token configured" path remove it with ``monkeypatch.delenv``.
"""

import pytest

from auth_helpers import TEST_TOKEN


@pytest.fixture(autouse=True)
def _hyperion_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure the shared test token in the environment for each test."""
    monkeypatch.setenv("HYPERION_API_TOKEN", TEST_TOKEN)
