"""
notion_client.py — Shared Notion helper for ~/ai/agents.

Wraps the official notion-client library with convenience methods for the
common patterns used across agents (read page, append block, query DB).

Role in the system
------------------
This module is the canonical entry point for any agent in ``~/ai/agents`` that
needs to read from or write to Notion (per the ``~/ai`` conventions: "Notion
reads in scripts use ``agents/_shared/notion_client.py``"). It is consumed by,
e.g., the secondbrain Notion mirror script. Centralizing access
here keeps authentication, pagination, and the Notion block/property JSON shape
in one place rather than re-implemented per script.

Key design decisions / non-obvious context
------------------------------------------
- Authentication is sourced from the ``NOTION_API_KEY`` environment variable by
  default (loaded from a gitignored ``.env``); callers may override it
  explicitly. Construction fails fast if no key is available.
- ``query_database`` transparently follows Notion's cursor pagination so callers
  always receive the full result set (Notion caps each response at 100 rows).
- Notion's REST payloads are deeply nested ("rich text" arrays, typed property
  objects). The helpers here build/parse just enough of that structure for the
  common agent use cases; they are intentionally minimal, not a full SDK wrapper.
- Page creation assumes the target database's title property is named "Name",
  which is the Notion default. Databases with a renamed title property will not
  match and the create call will fail server-side.

Usage:
    from _shared.notion_client import NotionHelper
    n = NotionHelper()
    pages = n.query_database("your-db-id")
"""

import os
from notion_client import Client


class NotionHelper:
    """Thin convenience wrapper around the official ``notion_client.Client``.

    Holds a single authenticated Notion client and exposes the handful of
    operations agents repeatedly need: paginated database queries, title
    extraction, appending text to a page, and creating database pages.

    Construct once and reuse; the underlying client is stateless beyond auth.

    Attributes:
        client: The authenticated ``notion_client.Client`` used for all calls.
    """

    def __init__(self, api_key: str | None = None):
        """Initialize the helper with a Notion integration token.

        Args:
            api_key: Notion integration token. If omitted, falls back to the
                ``NOTION_API_KEY`` environment variable.

        Raises:
            ValueError: If no API key is provided and ``NOTION_API_KEY`` is unset.
        """
        # Prefer an explicit key; otherwise read from the environment.
        key = api_key or os.environ.get("NOTION_API_KEY")
        if not key:
            raise ValueError("NOTION_API_KEY not set")
        self.client = Client(auth=key)

    def query_database(self, database_id: str, filter_: dict | None = None) -> list[dict]:
        """Return all pages from a database, handling pagination.

        Repeatedly calls the Notion ``databases.query`` endpoint, following the
        ``next_cursor`` until ``has_more`` is false, so the caller receives the
        complete result set rather than just the first page of 100 rows.

        Args:
            database_id: The Notion database ID to query.
            filter_: Optional Notion filter object (the ``filter`` payload as
                documented by the Notion API). When omitted, all pages are
                returned.

        Returns:
            A list of raw Notion page objects (dicts) across all result pages.
        """
        pages, cursor = [], None
        # `page_size=100` is Notion's per-request maximum; loop to drain cursors.
        while True:
            kwargs: dict = {"database_id": database_id, "page_size": 100}
            if filter_:
                kwargs["filter"] = filter_
            if cursor:
                kwargs["start_cursor"] = cursor
            result = self.client.databases.query(**kwargs)
            pages.extend(result["results"])
            # Stop once Notion reports there are no further pages to fetch.
            if not result.get("has_more"):
                break
            cursor = result["next_cursor"]
        return pages

    def get_page_title(self, page: dict) -> str:
        """Extract the plain-text title from a Notion page object.

        Scans the page's properties for the one whose type is ``"title"``
        (the property name varies per database, so we match on type, not name)
        and returns its first rich-text segment's plain text.

        Args:
            page: A raw Notion page object, e.g. an item from
                :meth:`query_database`.

        Returns:
            The page title as plain text, or ``"Untitled"`` if no title
            property is present or the title is empty.
        """
        # Notion titles can live under any property name; identify by type.
        for prop in page.get("properties", {}).values():
            if prop.get("type") == "title":
                rt = prop.get("title", [])
                # `rt` is a rich-text array; the first segment holds the text.
                return rt[0].get("plain_text", "Untitled") if rt else "Untitled"
        return "Untitled"

    def append_text(self, page_id: str, text: str) -> None:
        """Append a single paragraph block to an existing page.

        Args:
            page_id: The Notion page (or block) ID to append the child block to.
            text: The plain-text content of the new paragraph.

        Returns:
            None.

        Side effects:
            Mutates the target Notion page by adding a paragraph block.
        """
        # Build the minimal Notion block payload for a plain-text paragraph.
        self.client.blocks.children.append(
            block_id=page_id,
            children=[{
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": text}}]
                },
            }],
        )

    def create_page(self, database_id: str, title: str, content: str = "") -> dict:
        """Create a new page in a database with optional body content.

        The body, if provided, is split on blank lines (``"\\n\\n"``) into one
        Notion paragraph block per chunk. The page's title is set on the "Name"
        property, which is the default title property name for Notion databases.

        Args:
            database_id: The Notion database to create the page in.
            title: Title for the new page (written to the "Name" property).
            content: Optional body text; paragraphs are separated by blank lines.

        Returns:
            The raw Notion page object returned by the create API.

        Side effects:
            Creates a new page in the target Notion database.
        """
        children = []
        if content:
            # Each double-newline-delimited chunk becomes its own paragraph block.
            for para in content.split("\n\n"):
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": para.strip()}}]
                    },
                })
        return self.client.pages.create(
            parent={"database_id": database_id},
            properties={
                "Name": {"title": [{"type": "text", "text": {"content": title}}]}
            },
            children=children,
        )
