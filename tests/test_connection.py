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


def test_backup_keeps_the_original_settings_and_bad_shapes_are_refused(tmp_path):
    # Verifier 11b: a second click overwrote the .bak with already-modified settings; [] crashed with AttributeError.
    path = tmp_path / "claude_desktop_config.json"
    path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
    backup = connection.add_to_claude_desktop("a", path)
    connection.add_to_claude_desktop("b", path)
    assert json.loads(backup.read_text(encoding="utf-8")) == {"theme": "dark"}     # still the original
    for bad in ("[]", '{"mcpServers": null}'):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            connection.add_to_claude_desktop("a", path)
        assert path.read_text(encoding="utf-8") == bad



def test_stale_pid_file_never_kills_an_unrelated_process(tmp_path, monkeypatch):
    from app import connection

    monkeypatch.setattr(connection, "PID_FILE", tmp_path / "connection.pid")
    (tmp_path / "connection.pid").write_text("4242")
    calls = []
    monkeypatch.setattr(connection, "_is_our_server", lambda pid: False)
    monkeypatch.setattr(connection.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(connection.os, "kill", lambda *a: calls.append(a), raising=False)
    connection.stop_previous()
    assert calls == [] and not (tmp_path / "connection.pid").exists()
