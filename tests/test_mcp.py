"""MCP server: tools are listed as read-only and return the same JSON view as the API (fake LLM, toy catalog)."""

import asyncio
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from app.mcp_server import build_server
from test_pipeline import gen, resources


def call(server, name, args):
    return asyncio.run(server.call_tool(name, args))


def test_tools_are_read_only_and_match_spec_names(tmp_path):
    server = build_server(lambda: resources(tmp_path, []))
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    assert set(tools) == {"search_products", "get_similar"}
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools.values())
    assert {"query", "max_price", "category", "brand", "n"} <= set(tools["search_products"].input_schema["properties"])


def test_search_and_similar_tools(tmp_path):
    parsed = json.dumps({"semantic_query": "ice maker"})
    server = build_server(lambda: resources(tmp_path, [parsed, gen(["A"], "One GE ice maker.")]))
    out = call(server, "search_products", {"query": "ice maker", "brand": "GE", "n": 2})
    body = out.structured_content
    assert body["parsed"]["brand"] == "GE" and [p["product_id"] for p in body["products"]] == ["A", "C"]
    sim = call(server, "get_similar", {"product_id": "A", "n": 1}).structured_content
    assert sim["similar"][0]["product_id"] == "B"
    with pytest.raises(ToolError, match="unknown product_id zzz") as err:   # protocol layer turns it into is_error
        call(server, "get_similar", {"product_id": "zzz"})
    assert not isinstance(err.value, UnexpectedToolError)
    with pytest.raises(ToolError, match="query is empty"):
        call(server, "search_products", {"query": "  "})
