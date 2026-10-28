"""Credential hygiene shared by controlr and dojo. Standard library only: the agentic perceiver's stdlib client
(``dojo.perception.agentic.llm.UrllibChat``) runs in interpreters without httpx, so this module must not import
``controlr.llm``.

    redact(text, patterns=True, extra=())  every registered secret (exact value) and, with ``patterns``, every
                          credential-shaped substring -> ``<redacted>``: bearer / basic credentials, ``sk-...`` /
                          ``sk-ant-...`` / ``sk-or-...`` tokens, ``x-api-key`` / ``api_key`` / ``authorization`` /
                          ``*_token`` / ``secret`` / ``password`` values, URL userinfo, key / token query values
    clip(text, limit)     redact, THEN cut to ``limit`` characters (the other order can keep a key fragment that
                          straddles the cut)
    safe_label(value)     a server-reported id (served model, provider, finish reason) fit for a log: redacted, then
                          ``[A-Za-z0-9._:/@+-]{1,120}``, else ``<invalid>``
    register(*values)     exact values to redact from now on in this process (a client's own key, URL credentials)
    router_credentials(path, path_var)   (base URL, key) from ONE source: the dotenv file named by ``path_var``, or
                          the environment when no file is named; never a merge, no implicit home-dir fallback
    child_env(extra, model_calls=False)  the environment for a child process: the parent's minus every credential
                          name (see ``credential_name``); ``model_calls=True`` keeps only the router credential source

Redaction patterns and ``safe_label`` follow robot-harness ``tools/perceiver/keyguard.py`` (SEC verification,
robot-harness docs/notes/SEC-VERIFICATION-2026-10-28.md), adapted: no key-file policy here (ours: one named file or
the environment), plus a ``Basic`` scheme and JSON-escaped quotes.
"""
from __future__ import annotations

import os
import re
import urllib.parse
from pathlib import Path

R = "<redacted>"
ROUTER_KEYS = ("OMNIROUTE_BASE_URL", "OMNIROUTE_API_KEY")
SOURCE_VARS = ("AGENTIC_ENV_FILE", "DOJO_AGENT_DOTENV")     # name the dotenv file a model-calling process reads
MODEL_ENV = ROUTER_KEYS + SOURCE_VARS                       # what a model-calling child may inherit
# never inherited by a child (on top of the name rule below): the router's settings and the files that hold them,
# other providers' keys and base URLs (an agent session's, a developer shell's)
DENY_NAMES = frozenset(MODEL_ENV + ("OPENROUTER_API_KEY", "CONTROLR_CODEX_AUTH"))
DENY_PREFIXES = ("OMNIROUTE_", "OPENROUTER_", "ANTHROPIC_", "OPENAI_", "CLAUDE_CODE_")
# a name with one of these words (split on non-alphanumerics) is credential-bearing: OPENROUTER_API_KEY, HF_TOKEN,
# GITHUB_TOKEN, AWS_SECRET_ACCESS_KEY, SSH_AUTH_SOCK, CONTROLR_CODEX_AUTH, x-omniroute-api-key (a header) ...
# "TOKENS" is not "TOKEN": MAX_TOKENS, anthropic-ratelimit-tokens-remaining stay.
SECRET_WORDS = frozenset({"KEY", "APIKEY", "AUTHKEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL",
                          "CREDENTIALS", "COOKIE", "BEARER"})
# HMAC keys of OUR localhost services (multiprocessing.connection): per-run random values, and the children that
# serve or call those services need them, so they pass through child_env
LOCAL_AUTHKEYS = frozenset({"SIMLAB_TWIN_AUTHKEY", "SIMLAB_RENDER_AUTHKEY", "DOJO_PERCEPTION_AUTHKEY",
                            "HARNESS_SHIM_AUTHKEY"})

_SECRETS: set[str] = set()
_Q = r"[\"'\\]*"          # quotes and JSON escapes around a name or a value
_PATTERNS = (
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{6,}"), r"\1" + R),
    (re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{6,}"), R),
    (re.compile(r"(?i)(?<![a-z0-9])((?:x-api-key|api[_-]?key|proxy-authorization|authorization|access[_-]?token|"
                r"auth[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|secret|password|passwd)"
                + _Q + r"\s*[:=]\s*" + _Q + r"(?:(?:basic|bearer)\s+)?)(?!<redacted>)[^\s\"',;&}\\]+"), r"\1" + R),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+(?::[^/\s@]*)?@"), r"\1" + R + "@"),
    (re.compile(r"(?i)([?&](?:key|api[_-]?key|token|access[_-]?token|auth|sig|signature|secret)=)[^&\s#\"']+"),
     r"\1" + R),
)
_LABEL = re.compile(r"[A-Za-z0-9._:/@+-]{1,120}")


def register(*values) -> None:
    """Exact values to redact from now on in this process. Values shorter than 8 characters are ignored (they would
    redact ordinary words)."""
    _SECRETS.update(v for v in values if isinstance(v, str) and len(v) >= 8)


def url_secrets(url: str) -> list[str]:
    """The credential parts of a URL (userinfo, query values), for register()."""
    try:
        u = urllib.parse.urlsplit(url or "")
        user, pw = u.username, u.password
    except ValueError:
        return []
    return [x for x in (user, pw) if x] + [v for _, v in urllib.parse.parse_qsl(u.query) if v]


def redact(text, patterns: bool = True, extra=()):
    """``text`` with every registered secret, every value in ``extra`` and (``patterns``) every credential-shaped
    substring replaced by ``<redacted>``. Non-strings and empty strings come back unchanged."""
    if not isinstance(text, str) or not text:
        return text
    for s in sorted(_SECRETS.union(v for v in extra if isinstance(v, str) and len(v) >= 8), key=len, reverse=True):
        text = text.replace(s, R)
    if patterns:
        for rx, rep in _PATTERNS:
            text = rx.sub(rep, text)
    return text


def clip(text, limit: int, extra=()):
    """Redact the WHOLE text, then cut it to ``limit`` characters (+ "..."). Redacting after the cut would miss a key
    that starts just before it: only its head would remain, too short for a pattern and unequal to the key."""
    text = redact(text, extra=extra)
    if isinstance(text, str) and len(text) > limit:
        text = text[:limit] + "..."
    return text


def safe_label(value):
    """A server-chosen identifier for a log line: None stays None; a value that carries a credential -> <redacted>;
    anything outside ``[A-Za-z0-9._:/@+-]{1,120}`` -> <invalid>."""
    if value is None:
        return None
    if not isinstance(value, str):
        return "<invalid>"
    if redact(value) != value:
        return R
    return value if _LABEL.fullmatch(value) else "<invalid>"


def credential_name(name: str) -> bool:
    """A variable or header name that carries a credential (or names the file that holds one)."""
    n = name.upper()
    if n in DENY_NAMES or n.startswith(DENY_PREFIXES):
        return True
    return bool(SECRET_WORDS.intersection(w for w in re.split(r"[^A-Z0-9]+", n) if w))


# ------------------------------------------------------------------------------------------------- the router key

def read_dotenv(path, names) -> dict:
    """Only ``names`` from a dotenv file (``NAME=value`` lines, optional ``export``, quotes stripped); nothing else
    in the file is kept."""
    out = {}
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k.startswith("export "):
            k = k[len("export "):].strip()
        if k in names and k not in out:
            out[k] = v.strip().strip('"').strip("'")
    return out


def router_credentials(path: str | None, path_var: str, env=None) -> tuple[str, str]:
    """(OMNIROUTE_BASE_URL, OMNIROUTE_API_KEY) from ONE source per process:

    * ``path`` set (the file named by ``path_var``): both values from that file and nothing else; with OMNIROUTE_*
      also set in the environment the call is refused (no URL from one source and the key from the other);
    * no file named: both from the environment.

    No implicit fallback to a home-dir ``.env``. The key (and any URL credentials) are registered for redact().
    Errors name variables and paths, never values."""
    env = os.environ if env is None else env
    in_env = [k for k in ROUTER_KEYS if env.get(k)]
    if path:
        if in_env:
            raise RuntimeError(f"{path_var} names a key file and {', '.join(in_env)} is set in the environment: one "
                               f"source per process (unset {' / '.join(in_env)} or {path_var})")
        p = Path(path).expanduser()
        if not p.is_file():
            raise RuntimeError(f"{path_var}={p}: no such file")
        vals, src = read_dotenv(p, ROUTER_KEYS), f"{path_var}={p}"
    else:
        vals, src = {k: env[k] for k in in_env}, "the environment"
    missing = [k for k in ROUTER_KEYS if not vals.get(k)]
    if missing:
        raise RuntimeError(f"{' / '.join(missing)} not set in {src} (name a key file with {path_var}, or export both "
                           f"{' and '.join(ROUTER_KEYS)}; never both sources)")
    url, key = vals["OMNIROUTE_BASE_URL"], vals["OMNIROUTE_API_KEY"]
    register(key, *url_secrets(url))
    return url, key


# ------------------------------------------------------------------------------------------------ child processes

def child_env(extra: dict | None = None, *, model_calls: bool = False, base=None) -> dict:
    """Environment for a child process: ``base`` (default ``os.environ``) without credential-bearing names, then
    ``extra`` on top (explicit values, e.g. the authkey of the service this child serves, are always kept).

    Denylist, not allowlist: the children need PATH, HOME, CUDA / EGL / DISPLAY, PYTHON*, RD_* / DOJO_* / SIMLAB_* /
    J1_* settings and whatever RoboDojo's and Isaac's launchers read, a set no allowlist here would keep up with.
    Dropped: ``credential_name`` names (the router's OMNIROUTE_* and the files that hold them, AGENTIC_ENV_FILE /
    DOJO_AGENT_DOTENV, OPENROUTER_* / ANTHROPIC_* / OPENAI_*, and any name with a KEY / TOKEN / SECRET / PASSWORD /
    AUTH / COOKIE word) except our localhost authkeys (LOCAL_AUTHKEYS), and any variable whose value holds a
    registered secret. ``model_calls=True`` (a child that calls the model itself) keeps MODEL_ENV, nothing else."""
    src = os.environ if base is None else base
    keep = set(MODEL_ENV) if model_calls else set()
    out = {}
    for k, v in src.items():
        if k in keep or k in LOCAL_AUTHKEYS:
            out[k] = v
        elif credential_name(k) or (_SECRETS and redact(v, patterns=False) != v):
            continue
        else:
            out[k] = v
    out.update({str(k): str(v) for k, v in (extra or {}).items()})
    return out


__all__ = ["R", "ROUTER_KEYS", "SOURCE_VARS", "MODEL_ENV", "LOCAL_AUTHKEYS", "register", "url_secrets", "redact",
           "clip", "safe_label", "credential_name", "read_dotenv", "router_credentials", "child_env"]
