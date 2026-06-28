"""HTTP tests for the post-run follow-up chat endpoint (``POST /tasks/{id}/chat``).

Exercised in-process via ASGI transport against a patched ``tasks_dir``; the background
runner and the (LLM-backed) follow-up chat are mocked so no model is called.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from hyperion.server.api import _update_task, app


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def _submit(client):
    resp = await client.post("/tasks", json={"task": "research widgets", "hitl": "off"})
    return resp.json()["task_id"]


@pytest.mark.anyio
async def test_chat_on_done_task_returns_reply(tmp_path):
    """A follow-up on a done task returns the grounded reply and persists both turns."""
    from hyperion.config import settings

    with patch.object(settings, "tasks_dir", tmp_path):
        with patch("hyperion.server.api._run_and_update", new=AsyncMock()):
            with patch(
                "hyperion.followup.run_followup_chat", return_value="grounded answer"
            ) as mock_chat:
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    task_id = await _submit(client)
                    await _update_task(task_id, status="done")

                    resp = await client.post(
                        f"/tasks/{task_id}/chat", json={"message": "what did you find?"}
                    )
                    assert resp.status_code == 200
                    assert resp.json()["reply"] == "grounded answer"
                    mock_chat.assert_called_once()

                    # A second turn sees the first turn in its history argument.
                    resp2 = await client.post(
                        f"/tasks/{task_id}/chat", json={"message": "tell me more"}
                    )
                    assert resp2.status_code == 200
                    history_arg = mock_chat.call_args.args[2]
                    assert any(t["content"] == "what did you find?" for t in history_arg)


@pytest.mark.anyio
async def test_chat_on_unfinished_task_returns_409(tmp_path):
    """Chat is rejected (409) while the task is still queued/running."""
    from hyperion.config import settings

    with patch.object(settings, "tasks_dir", tmp_path):
        with patch("hyperion.server.api._run_and_update", new=AsyncMock()):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                task_id = await _submit(client)  # stays 'queued' (runner mocked)
                resp = await client.post(f"/tasks/{task_id}/chat", json={"message": "hi"})
                assert resp.status_code == 409


@pytest.mark.anyio
async def test_chat_on_missing_task_returns_404(tmp_path):
    """Chat on an unknown task id returns 404."""
    from hyperion.config import settings

    with patch.object(settings, "tasks_dir", tmp_path):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/tasks/nope/chat", json={"message": "hi"})
            assert resp.status_code == 404


@pytest.mark.anyio
async def test_chat_rejects_empty_message(tmp_path):
    """An empty/whitespace message is a 422."""
    from hyperion.config import settings

    with patch.object(settings, "tasks_dir", tmp_path):
        with patch("hyperion.server.api._run_and_update", new=AsyncMock()):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                task_id = await _submit(client)
                await _update_task(task_id, status="done")
                resp = await client.post(f"/tasks/{task_id}/chat", json={"message": "   "})
                assert resp.status_code == 422
