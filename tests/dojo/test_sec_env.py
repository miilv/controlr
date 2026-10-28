"""SEC on the dojo side (robot-harness docs/notes/SEC-VERIFICATION-2026-10-28.md, "Ilia-side changes proposed" 1-4):

* a provider error body echoing ``Authorization: Bearer <key>`` never reaches the agentic runner's transcript or stats
  (UrllibChat and the httpx OmniChat, a local 127.0.0.1 server), including a key that starts just before the cut;
* one key source per process: no home-dir ``.env`` fallback in llmio, no URL-from-one / key-from-another merge;
* children that never call the model (B1 worker, twin, agent tools) start without any credential name.

Fake keys only. A child writes the NAMES of its environment variables to a file, never a value."""

from __future__ import annotations

import http.server
import json
import os
import stat
import threading
from pathlib import Path

import pytest

from controlr import keyguard

FAKE = "sk-" + "ant-FAKE-0000-q7Wm2Lx9Rt4Yb8Nc3Vd6Hj1Kp5"   # split: CI's secret scan must not see a key shape
OPAQUE = "FAKEopaque0000zzzz1111qqqq2222wwww"       # matches no pattern: only exact-value redaction catches it
CRED_ENV = {"OMNIROUTE_BASE_URL": "http://127.0.0.1:9/v1", "OMNIROUTE_API_KEY": FAKE, "OPENROUTER_API_KEY": FAKE,
            "HF_TOKEN": "hf_FAKE0000", "OPENAI_API_KEY": FAKE, "SOME_SERVICE_PASSWORD": "FAKE0000"}
KEEP_ENV = {"RD_HOME": "/tmp/rd", "DOJO_SEC_PROBE": "1", "SIMLAB_SEC_PROBE": "1", "CUDA_VISIBLE_DEVICES": "0",
            "DOJO_PERCEPTION_AUTHKEY": "00ff"}


def fragments(key: str, text: str, n: int = 4) -> list[str]:
    return [key[i:i + n] for i in range(len(key) - n + 1) if key[i:i + n] in text]


def clean(text: str, key: str) -> bool:
    return not fragments(key, text) and "sk-a" not in text and "FAKE" not in text


def _envfile(tmp_path: Path, url: str, key: str) -> Path:
    f = tmp_path / "keys" / "agentic.env"
    f.parent.mkdir(exist_ok=True)
    f.write_text(f"OMNIROUTE_BASE_URL={url}\nOMNIROUTE_API_KEY={key}\n")
    return f


# ---------------------------------------------------------------------------
# finding 1: error bodies into the transcript / stats
# ---------------------------------------------------------------------------

class _Echo(http.server.BaseHTTPRequestHandler):
    """A gateway that echoes our Authorization header in a 400 body."""
    mode = "echo400"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        auth = self.headers["Authorization"]
        if self.mode == "echo400":
            body = json.dumps({"error": {"message": "bad request", "request_headers": {"Authorization": auth}}})
        elif self.mode == "pad400":   # the key starts at character ~290: the old 300-byte cut kept its head
            body = "x" * (290 - len("Authorization: Bearer ")) + f"Authorization: {auth} trailing text" + "." * 100
        else:                         # raw400: the bare key, no "Bearer", no JSON (only exact-value redaction catches it)
            body = "y" * 285 + " key " + auth.split(" ", 1)[1] + " " + "." * 100
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def echo_server():
    servers = []

    def start(mode: str) -> str:
        h = type("H", (_Echo,), {"mode": mode})
        srv = http.server.HTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}/v1"
    yield start
    for s in servers:
        s.shutdown()


@pytest.mark.parametrize("client", ["urllib", "httpx"])
@pytest.mark.parametrize("mode", ["echo400", "pad400", "raw400"])
def test_error_body_never_reaches_transcript_or_stats(tmp_path, monkeypatch, echo_server, client, mode):
    from dojo.perception.agentic import llm
    from dojo.perception.agentic.runner import run_agent

    key = OPAQUE if mode == "raw400" else FAKE
    for k in keyguard.ROUTER_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AGENTIC_ENV_FILE", str(_envfile(tmp_path, echo_server(mode), key)))
    chat = llm.UrllibChat(timeout_s=10) if client == "urllib" else llm.OmniChat(timeout_s=10, max_retries=0)
    r = chat("m", [{"role": "user", "content": "x"}], max_tokens=5)
    assert r.error.startswith("HTTP 400") and clean(r.error, key)
    stats = run_agent(chat, "m", "system", [{"type": "text", "text": "hi"}], tools=None, max_turns=3,
                      log_dir=tmp_path / "log")
    chat.close()
    transcript = (tmp_path / "log" / "transcript.jsonl").read_text()
    assert stats["stop"] == "llm_error" and len(stats["errors"]) == 2 and "HTTP 400" in transcript
    assert clean(transcript, key) and clean(json.dumps(stats), key)


def test_transcript_text_and_finish_reason_are_filtered(tmp_path):
    """The runner's one redaction step also covers the reply text (registered keys) and the finish reason."""
    from dojo.perception.agentic.llm import ChatReply
    from dojo.perception.agentic.runner import run_agent

    keyguard.register(OPAQUE)
    replies = [ChatReply(text=f'{{"answer": {{"objects": [], "note": "{OPAQUE}"}}}}', finish_reason=FAKE)]
    run_agent(lambda *a, **k: replies.pop(0), "m", "s", [{"type": "text", "text": "hi"}], tools=None,
              log_dir=tmp_path)
    t = (tmp_path / "transcript.jsonl").read_text()
    assert clean(t, OPAQUE) and clean(t, FAKE) and '"finish": "<redacted>"' in t


# ---------------------------------------------------------------------------
# findings 3-4: one source per process, no home-dir fallback
# ---------------------------------------------------------------------------

def test_llmio_never_reads_a_home_dir_env(tmp_path, monkeypatch):
    from dojo.agent import llmio

    for rel in ("controlr/.env", "skoltech/controlr/.env"):       # the two old fallbacks, under a fake HOME
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"OMNIROUTE_BASE_URL=https://home.test/v1\nOMNIROUTE_API_KEY={OPAQUE}\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    for k in keyguard.ROUTER_KEYS + ("DOJO_AGENT_DOTENV",):
        monkeypatch.delenv(k, raising=False)
    read = []
    real = keyguard.read_dotenv
    monkeypatch.setattr(keyguard, "read_dotenv", lambda p, names: read.append(str(p)) or real(p, names))
    with pytest.raises(RuntimeError, match="DOJO_AGENT_DOTENV") as e:
        llmio.omniroute_env()
    assert read == [] and OPAQUE not in str(e.value) and "home.test" not in str(e.value)
    # an explicitly named file is the one source
    named = _envfile(tmp_path, "https://named.test/v1", OPAQUE)
    monkeypatch.setenv("DOJO_AGENT_DOTENV", str(named))
    assert llmio.omniroute_env() == {"OMNIROUTE_BASE_URL": "https://named.test/v1", "OMNIROUTE_API_KEY": OPAQUE}
    assert read == [str(named)]
    # ... and the environment as well is refused
    monkeypatch.setenv("OMNIROUTE_API_KEY", "FAKEenv0000aaaa")
    with pytest.raises(RuntimeError, match="one source"):
        llmio.make_client()


def test_agentic_credentials_are_never_merged(tmp_path, monkeypatch):
    from dojo.perception.agentic.llm import router_credentials

    for k in keyguard.ROUTER_KEYS:
        monkeypatch.delenv(k, raising=False)
    key_only = tmp_path / "key_only.env"
    key_only.write_text(f"OMNIROUTE_API_KEY={OPAQUE}\n")
    monkeypatch.setenv("AGENTIC_ENV_FILE", str(key_only))
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "https://elsewhere.test/v1")   # the old code took this URL with the file's key
    with pytest.raises(RuntimeError, match="one source"):
        router_credentials()
    monkeypatch.delenv("OMNIROUTE_BASE_URL")
    with pytest.raises(RuntimeError, match="OMNIROUTE_BASE_URL not set"):
        router_credentials()
    monkeypatch.setenv("AGENTIC_ENV_FILE", str(_envfile(tmp_path, "https://file.test/v1", OPAQUE)))
    assert router_credentials() == ("https://file.test/v1", OPAQUE)
    monkeypatch.delenv("AGENTIC_ENV_FILE")
    monkeypatch.setenv("OMNIROUTE_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("OMNIROUTE_API_KEY", OPAQUE)
    assert router_credentials() == ("https://env.test/v1", OPAQUE)


# ---------------------------------------------------------------------------
# finding 6: child environments
# ---------------------------------------------------------------------------

def _names_probe(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """An executable that writes the NAMES of its environment variables to <out> and exits 3."""
    out = tmp_path / f"{name}.names"
    exe = tmp_path / f"{name}.sh"
    exe.write_text(f"#!/usr/bin/env bash\nenv -0 | tr '\\0' '\\n' | cut -d= -f1 | sort -u > '{out}'\nexit 3\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe, out


def _leaked(names: set[str]) -> list[str]:
    return sorted(n for n in names if keyguard.credential_name(n) and n not in keyguard.LOCAL_AUTHKEYS)


@pytest.fixture
def cred_env(tmp_path, monkeypatch):
    for k, v in {**CRED_ENV, **KEEP_ENV}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("AGENTIC_ENV_FILE", str(tmp_path / "agentic.env"))
    monkeypatch.setenv("DOJO_AGENT_DOTENV", str(tmp_path / "agent.env"))
    return tmp_path


def _read_names(out: Path, proc=None) -> set[str]:
    if proc is not None:
        proc.wait(timeout=30)
    return set(out.read_text().split())


def test_b1_worker_env_has_no_credentials(cred_env, monkeypatch):
    from dojo.perception import j1_adapter

    exe, out = _names_probe(cred_env, "b1")
    monkeypatch.setenv("J1_ILIA_PYTHON", str(exe))
    monkeypatch.setenv("J1_ILIA_LOG", str(cred_env / "j1_worker.log"))
    monkeypatch.setattr(j1_adapter, "_PROC", None)
    names = _read_names(out, j1_adapter._worker())
    assert _leaked(names) == [] and set(KEEP_ENV) <= names and {"PATH", "DOJO_CATALOG", "PYTHONPATH"} <= names


def test_agentic_worker_keeps_only_the_credential_source(cred_env, monkeypatch):
    from dojo.perception.agentic import j1

    exe, out = _names_probe(cred_env, "agentic")
    monkeypatch.setenv("J1_ILIA_PYTHON", str(exe))
    monkeypatch.setenv("AGENTIC_WORKER_LOG", str(cred_env / "agentic_worker.log"))
    monkeypatch.setattr(j1, "_PROC", None)
    names = _read_names(out, j1._worker())
    assert _leaked(names) == sorted(keyguard.MODEL_ENV) and set(KEEP_ENV) <= names


def test_twin_env_has_no_credentials_but_its_authkey(cred_env):
    from dojo.bridge.twin import TwinProcess

    exe, out = _names_probe(cred_env, "twin")
    with pytest.raises(RuntimeError, match="twin service exited"):
        TwinProcess(rdpy=str(exe), log_dir=cred_env / "twin", ready_timeout_s=30).start("stack_blocks")
    names = _read_names(out)
    assert _leaked(names) == [] and {"SIMLAB_TWIN_AUTHKEY", "SIMLAB_TWIN_PORT"} <= names and set(KEEP_ENV) <= names


def test_agent_tools_env_has_no_credentials(cred_env, monkeypatch):
    import subprocess

    from dojo.perception.agentic import tools_rd
    from dojo.perception.agentic.libero import LiberoTools

    exe, out = _names_probe(cred_env, "libero_tool")
    LiberoTools(cred_env, cred_env, "cell", None, "python3", "harness")._sub([str(exe)])
    names = _read_names(out)
    assert _leaked(names) == [] and set(KEEP_ENV) <= names and "OMP_NUM_THREADS" in names

    seen = {}

    def fake_run(argv, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(argv, 0, "ok", "")
    monkeypatch.setattr(tools_rd.subprocess, "run", fake_run)
    tools_rd.RoboDojoTools(cred_env).run("ph.py x")
    assert _leaked(set(seen["env"])) == [] and set(KEEP_ENV) <= set(seen["env"])


def test_parent_env_is_untouched(cred_env):
    """child_env builds a copy: the model-calling parent keeps its own source."""
    keyguard.child_env()
    assert os.environ.get("AGENTIC_ENV_FILE") and os.environ.get("OMNIROUTE_API_KEY") == FAKE
