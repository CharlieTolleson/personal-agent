"""
Shared bearer-token fixtures for Hyperion's HTTP tests.

Every Hyperion route requires ``Authorization: Bearer $HYPERION_API_TOKEN``
(``hyperion.server.auth``). ``conftest.py`` sets ``HYPERION_API_TOKEN`` to
``TEST_TOKEN`` for every test, and API tests pass ``AUTH_HEADERS`` to their client.
The token is a fixed, obviously fake test value (64 chars, above the 32-char floor).
"""

TEST_TOKEN = "test-token-" + "0123456789abcdef" * 3 + "x" * 5
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_TOKEN}"}
