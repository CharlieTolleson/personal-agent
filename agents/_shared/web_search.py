"""
OWUI tool architecture note — not an importable module.

The canonical web search implementation lives in:
    agents/hyperion/src/hyperion/tools/web_search.py

It is exposed over HTTP at:
    GET http://localhost:4100/tools/search?q=<query>[&top_k=10][&categories=general,news]

OWUI tool plugins call that endpoint and are intentionally thin HTTP wrappers.
This means updating web search behavior (SearXNG params, reranking, sanitization)
only requires editing the Hyperion tool — the OWUI plugin picks up the change
automatically since it delegates to the same endpoint.

See ~/secondbrain/CLAUDE.md for the pinned reminder.
"""
