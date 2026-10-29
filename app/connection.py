"""Start and stop the website connection (the REST API in app/api.py) from the demo app, without a terminal."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PORT = 8000
URL = f"http://localhost:{PORT}"
PID_FILE = ROOT / "artifacts" / "connection.pid"      # lets the app stop a connection it started before a restart


def website_origin(address: str) -> str | None:
    """'mystore.com' or 'https://mystore.com/shop' -> 'https://mystore.com' (the browser origin to allow)."""
    address = address.strip()
    if not address:
        return None
    if "://" not in address:
        address = "https://" + address
    scheme, rest = address.split("://", 1)
    host = rest.split("/")[0]
    return f"{scheme.lower()}://{host.lower()}" if host else None


def is_running(timeout: float = 1.0) -> bool:
    try:
        return httpx.get(f"{URL}/health", timeout=timeout).json().get("status") == "ok"
    except (httpx.HTTPError, ValueError):
        return False


def start(origins: list[str], wait_s: float = 120.0) -> subprocess.Popen:
    """Launch the API on localhost:8000 allowing the given website origins; wait until it answers."""
    allowed = [URL, f"http://127.0.0.1:{PORT}"] + [o for o in origins if o]
    env = {**os.environ, "CONCIERGE_ALLOWED_ORIGINS": ",".join(allowed)}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1",
                             "--port", str(PORT)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("the connection could not start (is port 8000 used by another program?)")
        if is_running():
            PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            PID_FILE.write_text(str(proc.pid), encoding="utf-8")
            atexit.register(stop, proc)                 # closing the app also closes the connection
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("the connection did not start in time")


def stop(proc: subprocess.Popen | None) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    PID_FILE.unlink(missing_ok=True)


def started_by_app() -> bool:
    """True when the running connection was started by this app (now or before a restart)."""
    return PID_FILE.exists()


def stop_previous() -> None:
    """Stop a connection this app started in an earlier session (its process id is in PID_FILE)."""
    if not PID_FILE.exists():
        return
    pid = int(PID_FILE.read_text(encoding="utf-8").strip() or 0)
    if pid:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
    PID_FILE.unlink(missing_ok=True)


# ---- Claude Desktop (MCP) ----------------------------------------------------------------------

def claude_desktop_config_path() -> Path:
    """Where Claude Desktop keeps its settings on this computer."""
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return appdata / "Claude" / "claude_desktop_config.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def server_name(store: str | None) -> str:
    return f"shopping-assistant-{store or 'demo'}"


def claude_desktop_entry(store: str | None) -> dict:
    """MCP settings that start app/mcp_server.py for one store (no working directory needed)."""
    return {"mcpServers": {server_name(store): {
        "command": str(Path(sys.executable)),
        "args": [str(ROOT / "app" / "mcp_server.py")],
        "env": {"CONCIERGE_STORE": store or ""}}}}


def add_to_claude_desktop(store: str | None, path: Path) -> Path | None:
    """Add this store's MCP server to Claude Desktop's settings, keeping everything else.

    Returns the backup of the settings as they were before the first change (None when there were none); later
    clicks keep that original backup. Refuses settings that are not a JSON object, so nothing is ever lost.
    """
    settings: dict = {}
    backup = None
    if path.exists() and path.read_text(encoding="utf-8").strip():
        original = path.read_text(encoding="utf-8")
        try:
            settings = json.loads(original)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON; fix or remove it first") from exc
        if not isinstance(settings, dict) or not isinstance(settings.get("mcpServers", {}), dict):
            raise ValueError(f"{path} has an unexpected layout; fix it first")
        backup = path.with_name(path.name + ".bak")
        if not backup.exists():                         # keep the very first original
            backup.write_text(original, encoding="utf-8")
    settings.setdefault("mcpServers", {}).update(claude_desktop_entry(store)["mcpServers"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    os.replace(tmp, path)                               # atomic: never a half-written settings file
    return backup
