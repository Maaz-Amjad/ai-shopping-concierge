"""Website connection helpers and the Claude Desktop settings merge (never loses the user's settings)."""

import json

import pytest

from app import connection


def test_website_origin():
    assert connection.website_origin("MyStore.com/shop") == "https://mystore.com"
    assert connection.website_origin("http://localhost:3000/x") == "http://localhost:3000"
    assert connection.website_origin("  ") is None


def test_claude_desktop_settings_are_merged_and_backed_up(tmp_path):
    path = tmp_path / "Claude" / "claude_desktop_config.json"
    assert connection.add_to_claude_desktop(None, path) is None                 # fresh install: no backup needed
    path.write_text(json.dumps({"theme": "dark", "mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
    backup = connection.add_to_claude_desktop("mystore", path)
    settings = json.loads(path.read_text(encoding="utf-8"))
    assert settings["theme"] == "dark" and set(settings["mcpServers"]) == {"other", "shopping-assistant-mystore"}
    entry = settings["mcpServers"]["shopping-assistant-mystore"]
    assert entry["args"][0].endswith("mcp_server.py") and entry["env"] == {"CONCIERGE_STORE": "mystore"}
    assert json.loads(backup.read_text(encoding="utf-8"))["theme"] == "dark"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        connection.add_to_claude_desktop("mystore", path)
    assert path.read_text(encoding="utf-8") == "{not json"                      # left untouched
