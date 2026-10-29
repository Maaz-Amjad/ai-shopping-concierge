"""Read-only MCP server (spec 11 / FR-U3): AI shopping agents can search a store and find similar products.

Run (stdio): .venv/Scripts/python.exe app/mcp_server.py            store = $CONCIERGE_STORE (default: benchmark)
Claude Desktop: the demo app's "Connect to Claude Desktop" button writes the settings (app/connection.py).
No write tools: nothing here can change a catalog, order, or cart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `python app/mcp_server.py` works from anywhere

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from src import load_config, pipeline
from src.pipeline import Resources, app_config, load_resources, records, result_to_dict

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


def build_server(loader: Callable[[], Resources] | None = None) -> MCPServer:
    """Create the server; `loader()` returns Resources and is called once, on the first tool call."""
    cfg = load_config()
    loader = loader or (lambda: load_resources(app_config(cfg), max_calls=cfg.get("app", {}).get("max_llm_calls"),
                                               store=os.environ.get("CONCIERGE_STORE") or None))
    cache: list[Resources] = []

    def res() -> Resources:
        if not cache:
            cache.append(loader())
        return cache[0]

    server = MCPServer(name="shopping-concierge", version="1.0", instructions=(
        "Search one store's catalog. Hard limits (price, category, brand) are enforced exactly; descriptive wishes "
        "go in the query text. Recommend only products returned by these tools and quote their prices exactly."))

    @server.tool(annotations=READ_ONLY)
    def search_products(query: str, max_price: float | None = None, min_price: float | None = None,
                        category: str | None = None, brand: str | None = None, n: int = 5) -> dict[str, Any]:
        """Find products for a natural-language request. Optional filters override what is parsed from the query;
        category and brand must be catalog values (unknown ones are dropped and reported in flags.parse_log).
        Returns the parsed filters, ranked products with prices and score components, and a grounded answer."""
        if not query.strip():
            raise ToolError("query is empty")
        r = res()
        out = pipeline.search_products(query, r, max_price=max_price, min_price=min_price, category=category,
                                       brand=brand, n=max(1, min(n, 20)))
        return result_to_dict(query, out, r)

    @server.tool(name="get_similar", annotations=READ_ONLY)
    def get_similar(product_id: str, n: int = 5) -> dict[str, Any]:
        """Products most similar to `product_id` (by description embedding), with prices and similarity."""
        try:
            similar = pipeline.get_similar(product_id, res(), max(1, min(n, 20)))
        except KeyError as exc:                     # anticipated: the agent sees the message, not a crash
            raise ToolError(str(exc).strip("'")) from None
        return {"product_id": product_id, "similar": records(similar)}

    return server


server = build_server()

if __name__ == "__main__":
    server.run("stdio")
