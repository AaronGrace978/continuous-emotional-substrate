"""
Multi-provider voice catalog for ChronoSoul.

Credentials come from the local settings file or from environment variables.
Nothing in this module hardcodes a secret. The desktop chamber reads
``chronosoul_providers.json`` (gitignored). The CLI bridge uses the environment.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


class ProviderError(RuntimeError):
    """A remote voice refused the request or was not configured."""


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    blurb: str
    kind: str  # local | openai | anthropic | ollama
    base_url: str
    default_model: str
    models: tuple
    env_key: str
    key_required: bool
    base_url_editable: bool
    chat_path: str = "/chat/completions"


def _spec(**kwargs) -> ProviderSpec:
    return ProviderSpec(**kwargs)


# Order is the order shown in Settings. Local PEMS stays first and is the default.
CATALOG: tuple = (
    _spec(
        id="local",
        label="Local PEMS",
        blurb="Qwen2.5-1.5B with the PEMS adapter on this machine",
        kind="local",
        base_url="",
        default_model="Qwen2.5-1.5B-Instruct + PEMS",
        models=("Qwen2.5-1.5B-Instruct + PEMS",),
        env_key="",
        key_required=False,
        base_url_editable=False,
        chat_path="",
    ),
    _spec(
        id="ollama",
        label="Ollama",
        blurb="A model served by a local Ollama daemon",
        kind="ollama",
        base_url="http://127.0.0.1:11434",
        default_model="llama3.2",
        models=(
            "llama3.2",
            "llama3.1",
            "qwen2.5:1.5b",
            "qwen2.5:7b",
            "qwen2.5:14b",
            "gemma3",
            "mistral",
            "phi4",
            "deepseek-r1",
        ),
        env_key="",
        key_required=False,
        base_url_editable=True,
        chat_path="/api/chat",
    ),
    _spec(
        id="ollama_cloud",
        label="Ollama Cloud",
        blurb="Ollama's hosted models at ollama.com",
        kind="ollama",
        base_url="https://ollama.com",
        default_model="gpt-oss:120b",
        models=(
            "gpt-oss:20b",
            "gpt-oss:120b",
            "qwen3-coder:480b",
            "deepseek-v3.1:671b",
            "kimi-k2:1t",
        ),
        env_key="OLLAMA_API_KEY",
        key_required=True,
        base_url_editable=False,
        chat_path="/api/chat",
    ),
    _spec(
        id="openai",
        label="OpenAI",
        blurb="GPT models via api.openai.com",
        kind="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        models=("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "o4-mini", "o3-mini"),
        env_key="OPENAI_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="anthropic",
        label="Anthropic",
        blurb="Claude via the Messages API",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-5",
        models=(
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-opus-4-1",
            "claude-3-5-haiku-latest",
            "claude-3-5-sonnet-latest",
        ),
        env_key="ANTHROPIC_API_KEY",
        key_required=True,
        base_url_editable=False,
        chat_path="/v1/messages",
    ),
    _spec(
        id="gemini",
        label="Gemini",
        blurb="Google Gemini, OpenAI-compatible endpoint",
        kind="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-2.5-flash",
        models=("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite"),
        env_key="GEMINI_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="groq",
        label="Groq",
        blurb="Fast hosted open models",
        kind="openai",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b", "qwen/qwen3-32b"),
        env_key="GROQ_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="mistral",
        label="Mistral",
        blurb="Mistral La Plateforme",
        kind="openai",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
        models=("mistral-small-latest", "mistral-large-latest", "open-mistral-nemo", "codestral-latest"),
        env_key="MISTRAL_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="openrouter",
        label="OpenRouter",
        blurb="One key for many hosted models",
        kind="openai",
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o-mini",
        models=(
            "openai/gpt-4o-mini",
            "anthropic/claude-sonnet-4.5",
            "google/gemini-2.5-flash",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
        ),
        env_key="OPENROUTER_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="together",
        label="Together",
        blurb="Together AI inference",
        kind="openai",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        models=(
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
        ),
        env_key="TOGETHER_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="fireworks",
        label="Fireworks",
        blurb="Fireworks AI inference",
        kind="openai",
        base_url="https://api.fireworks.ai/inference/v1",
        default_model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        models=(
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
        ),
        env_key="FIREWORKS_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="deepseek",
        label="DeepSeek",
        blurb="DeepSeek chat and reasoner",
        kind="openai",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        models=("deepseek-chat", "deepseek-reasoner"),
        env_key="DEEPSEEK_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="xai",
        label="xAI",
        blurb="Grok models",
        kind="openai",
        base_url="https://api.x.ai/v1",
        default_model="grok-3-mini",
        models=("grok-3-mini", "grok-3", "grok-2-latest"),
        env_key="XAI_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="cohere",
        label="Cohere",
        blurb="Cohere compatibility API",
        kind="openai",
        base_url="https://api.cohere.com/compatibility/v1",
        default_model="command-r-plus",
        models=("command-r", "command-r-plus", "command-a-03-2025"),
        env_key="COHERE_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="perplexity",
        label="Perplexity",
        blurb="Sonar models",
        kind="openai",
        base_url="https://api.perplexity.ai",
        default_model="sonar",
        models=("sonar", "sonar-pro", "sonar-reasoning"),
        env_key="PERPLEXITY_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="cerebras",
        label="Cerebras",
        blurb="Cerebras inference",
        kind="openai",
        base_url="https://api.cerebras.ai/v1",
        default_model="llama-3.3-70b",
        models=("llama3.1-8b", "llama-3.3-70b", "qwen-3-32b"),
        env_key="CEREBRAS_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="nvidia",
        label="NVIDIA",
        blurb="NVIDIA NIM / integrate API",
        kind="openai",
        base_url="https://integrate.api.nvidia.com/v1",
        default_model="meta/llama-3.3-70b-instruct",
        models=("meta/llama-3.1-8b-instruct", "meta/llama-3.3-70b-instruct", "google/gemma-2-9b-it"),
        env_key="NVIDIA_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="github",
        label="GitHub Models",
        blurb="Models on GitHub, using a GitHub token",
        kind="openai",
        base_url="https://models.github.ai/inference",
        default_model="gpt-4o-mini",
        models=("gpt-4o-mini", "gpt-4o", "Meta-Llama-3.1-70B-Instruct", "Phi-4"),
        env_key="GITHUB_TOKEN",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="sambanova",
        label="SambaNova",
        blurb="SambaNova Cloud",
        kind="openai",
        base_url="https://api.sambanova.ai/v1",
        default_model="Meta-Llama-3.3-70B-Instruct",
        models=("Meta-Llama-3.1-8B-Instruct", "Meta-Llama-3.3-70B-Instruct", "DeepSeek-V3-0324"),
        env_key="SAMBANOVA_API_KEY",
        key_required=True,
        base_url_editable=False,
    ),
    _spec(
        id="custom",
        label="Custom",
        blurb="Any other OpenAI-compatible endpoint",
        kind="openai",
        base_url="",
        default_model="",
        models=(),
        env_key="CHRONOSOUL_CUSTOM_API_KEY",
        key_required=False,
        base_url_editable=True,
    ),
)

BY_ID: Dict[str, ProviderSpec] = {spec.id: spec for spec in CATALOG}

# CLI auto-detect order when CHRONOSOUL_PROVIDER is unset. The chamber itself
# stays on Local PEMS until Settings says otherwise.
ENV_PRIORITY: tuple = (
    "openai",
    "anthropic",
    "gemini",
    "ollama_cloud",
    "groq",
    "mistral",
    "openrouter",
    "together",
    "fireworks",
    "deepseek",
    "xai",
    "cohere",
    "perplexity",
    "cerebras",
    "nvidia",
    "github",
    "sambanova",
    "custom",
    "ollama",
)

_EXTRA_ENV = {
    "gemini": ("GOOGLE_API_KEY",),
    "ollama": (),
}


def catalog_ids() -> List[str]:
    return [spec.id for spec in CATALOG]


def _clean(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:limit]


def _http_url(value: str) -> str:
    text = value.strip().rstrip("/")
    if not text:
        return ""
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ProviderError("Base URL must start with http:// or https://")
    return text


def endpoint_url(base_url: str, chat_path: str) -> str:
    """Join a base URL and a chat path without duplicating a path the user already included."""
    base = (base_url or "").strip().rstrip("/")
    path = chat_path if chat_path.startswith("/") else "/" + chat_path
    tail = path.rstrip("/")
    if base.endswith(tail) or base.endswith("/chat/completions") or base.endswith("/messages") or base.endswith("/api/chat"):
        return base
    return base + path


def anthropic_parts(messages: Sequence[dict]) -> tuple:
    """Split system text out of a chat list and merge adjacent turns of the same role."""
    system: List[str] = []
    turns: List[dict] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        if role == "system":
            if content:
                system.append(content)
            continue
        if role not in ("user", "assistant"):
            role = "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": role, "content": content})
    if not turns:
        turns = [{"role": "user", "content": ""}]
    if turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": "(continuing)"})
    return "\n\n".join(system), turns


def _env_key(spec: ProviderSpec) -> str:
    names = [spec.env_key] if spec.env_key else []
    names.extend(_EXTRA_ENV.get(spec.id, ()))
    if spec.id == "ollama":
        return ""
    for name in names:
        if not name:
            continue
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _ollama_host_from_env() -> str:
    for name in ("OLLAMA_ENDPOINT", "OLLAMA_HOST"):
        value = os.environ.get(name, "").strip()
        if value:
            return value.rstrip("/")
    return ""


class ProviderSettings:
    """Selected provider, per-provider model, optional base URL, and saved keys."""

    def __init__(self, path: Optional[Path] = None, data: Optional[dict] = None):
        self.path = path
        self.data = {
            "provider": "local",
            "models": {},
            "base_urls": {},
            "keys": {},
        }
        if data is not None:
            self.data = _merge(self.data, data)
        elif path is not None and path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                loaded = None
            if isinstance(loaded, dict):
                self.data = _merge(self.data, loaded)
        self._normalize()

    def spec(self) -> ProviderSpec:
        return BY_ID.get(self.data["provider"], BY_ID["local"])

    def uses_remote(self) -> bool:
        return self.spec().kind != "local"

    def model_for(self, provider_id: Optional[str] = None) -> str:
        spec = BY_ID.get(provider_id or self.data["provider"], self.spec())
        saved = _clean(self.data["models"].get(spec.id, ""), 200)
        if saved:
            return saved
        if spec.id == "ollama":
            env_model = _clean(os.environ.get("OLLAMA_MODEL", ""), 200)
            if env_model:
                return env_model
        env_model = _clean(os.environ.get("CHRONOSOUL_MODEL", ""), 200) if provider_id is None else ""
        if env_model and (provider_id is None or provider_id == self.data["provider"]):
            return env_model
        return spec.default_model

    def base_url_for(self, provider_id: Optional[str] = None) -> str:
        spec = BY_ID.get(provider_id or self.data["provider"], self.spec())
        saved = _clean(self.data["base_urls"].get(spec.id, ""), 300)
        if saved:
            return saved
        if spec.id == "ollama":
            host = _ollama_host_from_env()
            if host:
                return host
        if spec.id == "custom":
            return _clean(os.environ.get("CHRONOSOUL_CUSTOM_BASE_URL", ""), 300)
        return spec.base_url

    def api_key(self, provider_id: Optional[str] = None) -> str:
        spec = BY_ID.get(provider_id or self.data["provider"], self.spec())
        saved = _clean(self.data["keys"].get(spec.id, ""), 500)
        if saved:
            return saved
        return _env_key(spec)

    def key_source(self, provider_id: str) -> str:
        spec = BY_ID[provider_id]
        if _clean(self.data["keys"].get(provider_id, ""), 500):
            return "saved"
        if _env_key(spec):
            return "env"
        return ""

    def configured(self, spec: Optional[ProviderSpec] = None) -> bool:
        spec = spec or self.spec()
        if spec.kind == "local":
            return True
        if not self.model_for(spec.id):
            return False
        if spec.base_url_editable and not self.base_url_for(spec.id):
            return False
        if spec.key_required and not self.api_key(spec.id):
            return False
        return True

    def missing_reason(self, spec: Optional[ProviderSpec] = None) -> str:
        spec = spec or self.spec()
        if spec.kind == "local":
            return ""
        if spec.key_required and not self.api_key(spec.id):
            hint = f" Set {spec.env_key} or paste a key in Settings." if spec.env_key else " Paste a key in Settings."
            return f"{spec.label} needs an API key.{hint}"
        if spec.base_url_editable and not self.base_url_for(spec.id):
            return f"{spec.label} needs a base URL."
        if not self.model_for(spec.id):
            return f"{spec.label} needs a model id."
        return ""

    def annotate(self, model_info: dict) -> None:
        """Record the selected voice on the chamber status. Remote voices skip the local weights."""
        spec = self.spec()
        model = self.model_for(spec.id)
        model_info["provider"] = spec.id
        model_info["provider_label"] = spec.label
        model_info["model_name"] = model
        if spec.kind == "local":
            return
        model_info["device"] = "remote"
        model_info["steering"] = "off"
        reason = self.missing_reason(spec)
        if reason:
            model_info["status"] = "error"
            model_info["detail"] = reason
            return
        model_info["status"] = "ready"
        model_info["detail"] = f"{spec.label} · {model}"

    def public(self) -> dict:
        """Catalog plus the selection. Keys are never included."""
        selected = self.spec()
        providers = []
        for spec in CATALOG:
            providers.append({
                "id": spec.id,
                "label": spec.label,
                "blurb": spec.blurb,
                "kind": spec.kind,
                "models": list(spec.models),
                "default_model": spec.default_model,
                "model": self.model_for(spec.id),
                "key_required": spec.key_required,
                "key_optional": spec.kind != "local" and not spec.key_required,
                "env_key": spec.env_key,
                "has_key": bool(self.api_key(spec.id)) if spec.kind != "local" else False,
                "key_source": self.key_source(spec.id) if spec.kind != "local" else "",
                "base_url": self.base_url_for(spec.id),
                "base_url_editable": spec.base_url_editable,
            })
        return {
            "provider": selected.id,
            "model": self.model_for(selected.id),
            "base_url": self.base_url_for(selected.id),
            "providers": providers,
        }

    def update(self, payload: dict) -> None:
        if not isinstance(payload, dict):
            raise ProviderError("Settings must be an object.")
        provider_id = _clean(payload.get("provider"), 40)
        if provider_id not in BY_ID:
            raise ProviderError("Unknown provider.")
        spec = BY_ID[provider_id]
        model = _clean(payload.get("model"), 200) or spec.default_model
        if spec.id == "custom" and not model:
            raise ProviderError("Custom needs a model id.")
        base_url = self.base_url_for(spec.id)
        if spec.base_url_editable and "base_url" in payload:
            base_url = _http_url(_clean(payload.get("base_url"), 300)) if _clean(payload.get("base_url"), 300) else ""
        if spec.base_url_editable and spec.id != "ollama" and not base_url:
            raise ProviderError(f"{spec.label} needs a base URL.")
        if spec.id == "ollama" and not base_url:
            base_url = spec.base_url

        keys = dict(self.data["keys"])
        if payload.get("clear_key"):
            keys.pop(spec.id, None)
        incoming = payload.get("api_key")
        if isinstance(incoming, str) and incoming.strip():
            key = incoming.strip()
            if len(key) > 500 or any(ch in key for ch in "\r\n"):
                raise ProviderError("API key looks invalid.")
            if spec.kind == "local":
                raise ProviderError("The local voice does not take an API key.")
            keys[spec.id] = key

        self.data["provider"] = spec.id
        models = dict(self.data["models"])
        if model:
            models[spec.id] = model
        self.data["models"] = models
        urls = dict(self.data["base_urls"])
        if spec.base_url_editable:
            urls[spec.id] = base_url
        self.data["base_urls"] = urls
        self.data["keys"] = keys
        self._normalize()
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": self.data["provider"],
            "models": self.data["models"],
            "base_urls": self.data["base_urls"],
            "keys": self.data["keys"],
        }
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_bytes(raw)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(self.path)

    def stream(self, messages: Sequence[dict], cancel: Optional[threading.Event] = None,
               max_tokens: int = 2048, temperature: float = 0.7) -> Iterator[str]:
        spec = self.spec()
        if spec.kind == "local":
            raise ProviderError("The local PEMS voice is not a remote provider.")
        reason = self.missing_reason(spec)
        if reason:
            raise ProviderError(reason)
        cancel = cancel or threading.Event()
        model = self.model_for(spec.id)
        base = self.base_url_for(spec.id)
        key = self.api_key(spec.id)
        if spec.kind == "anthropic":
            yield from _stream_anthropic(base, spec.chat_path, key, model, messages, cancel, max_tokens, temperature)
            return
        if spec.kind == "ollama":
            yield from _stream_ollama(base, spec.chat_path, key, model, messages, cancel, max_tokens, temperature)
            return
        yield from _stream_openai(base, spec.chat_path, key, model, messages, cancel, max_tokens, temperature, spec)

    def complete(self, messages: Sequence[dict], cancel: Optional[threading.Event] = None,
                 max_tokens: int = 800) -> str:
        parts = []
        for piece in self.stream(messages, cancel=cancel, max_tokens=max_tokens, temperature=0.2):
            parts.append(piece)
        text = "".join(parts).strip()
        if cancel is not None and cancel.is_set() and not text:
            return ""
        return text

    def _normalize(self) -> None:
        if self.data.get("provider") not in BY_ID:
            self.data["provider"] = "local"
        for field in ("models", "base_urls", "keys"):
            raw = self.data.get(field)
            clean: Dict[str, str] = {}
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if key in BY_ID and isinstance(value, str):
                        clean[key] = value.strip()[:500]
            self.data[field] = clean


def _merge(base: dict, loaded: dict) -> dict:
    merged = {
        "provider": loaded.get("provider") or base["provider"],
        "models": dict(base["models"]),
        "base_urls": dict(base["base_urls"]),
        "keys": dict(base["keys"]),
    }
    for field in ("models", "base_urls", "keys"):
        extra = loaded.get(field)
        if isinstance(extra, dict):
            merged[field].update(extra)
    return merged


def settings_from_env() -> ProviderSettings:
    """In-memory selection for the CLI. The chamber does not use this."""
    chosen = _clean(os.environ.get("CHRONOSOUL_PROVIDER", ""), 40)
    if chosen == "simulated" or (chosen and chosen not in BY_ID):
        chosen = ""
    if not chosen:
        for provider_id in ENV_PRIORITY:
            spec = BY_ID[provider_id]
            if provider_id == "ollama" and (_ollama_host_from_env() or os.environ.get("OLLAMA_MODEL")):
                chosen = provider_id
                break
            if spec.env_key and _env_key(spec):
                chosen = provider_id
                break
    if not chosen:
        settings = ProviderSettings()
        settings.data["provider"] = "local"
        return settings
    settings = ProviderSettings(data={"provider": chosen})
    model = _clean(os.environ.get("CHRONOSOUL_MODEL", "") or os.environ.get("LLM_MODEL", ""), 200)
    if not model and chosen == "ollama":
        model = _clean(os.environ.get("OLLAMA_MODEL", ""), 200)
    if model:
        settings.data["models"][chosen] = model
    if chosen == "custom":
        base = _clean(os.environ.get("CHRONOSOUL_CUSTOM_BASE_URL", ""), 300)
        if base:
            settings.data["base_urls"]["custom"] = base
    return settings


def _request(url: str, payload: dict, headers: dict, timeout: int = 120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:400]
        except Exception:
            body = ""
        detail = _error_detail(body) or body or exc.reason
        raise ProviderError(f"The voice returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach the voice: {exc.reason}") from exc


def _error_detail(body: str) -> str:
    try:
        parsed = json.loads(body)
    except Exception:
        return ""
    if isinstance(parsed, dict):
        err = parsed.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err.get("type") or "")[:300]
        if isinstance(err, str):
            return err[:300]
        if parsed.get("message"):
            return str(parsed["message"])[:300]
    return ""


def _headers(key: str, accept: str) -> dict:
    headers = {"Content-Type": "application/json", "Accept": accept}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _iter_lines(resp, cancel: threading.Event) -> Iterable[str]:
    try:
        while not cancel.is_set():
            raw = resp.readline()
            if not raw:
                break
            yield raw.decode("utf-8", "replace").strip()
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _stream_openai(base: str, chat_path: str, key: str, model: str, messages: Sequence[dict],
                   cancel: threading.Event, max_tokens: int, temperature: float,
                   spec: ProviderSpec) -> Iterator[str]:
    url = endpoint_url(base, chat_path)
    payload = {
        "model": model,
        "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    headers = _headers(key, "text/event-stream")
    if spec.id == "openrouter":
        headers["X-Title"] = "ChronoSoul"
    resp = _request(url, payload, headers)
    yielded = False
    for line in _iter_lines(resp, cancel):
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            if data == "[DONE]":
                break
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("error"):
            raise ProviderError(_error_detail(data) or "The voice returned an error.")
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        if not choices:
            continue
        delta = choices[0].get("delta") or choices[0].get("message") or {}
        text = delta.get("content") or ""
        if text:
            yielded = True
            yield text
    if cancel.is_set():
        return
    if not yielded:
        return


def _stream_anthropic(base: str, chat_path: str, key: str, model: str, messages: Sequence[dict],
                      cancel: threading.Event, max_tokens: int, temperature: float) -> Iterator[str]:
    system, turns = anthropic_parts(messages)
    url = endpoint_url(base, chat_path)
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": turns,
        "stream": True,
    }
    if system:
        payload["system"] = system
    headers = _headers(key, "text/event-stream")
    headers["anthropic-version"] = "2023-06-01"
    resp = _request(url, payload, headers)
    for line in _iter_lines(resp, cancel):
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data:
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("type") == "error":
            err = parsed.get("error") or {}
            message = err.get("message") if isinstance(err, dict) else err
            raise ProviderError(str(message or "Anthropic returned an error."))
        delta = parsed.get("delta") or {}
        text = ""
        if isinstance(delta, dict):
            text = delta.get("text") or ""
        if text:
            yield text


def _stream_ollama(base: str, chat_path: str, key: str, model: str, messages: Sequence[dict],
                   cancel: threading.Event, max_tokens: int, temperature: float) -> Iterator[str]:
    url = endpoint_url(base, chat_path)
    payload = {
        "model": model,
        "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    headers = _headers(key, "application/x-ndjson")
    resp = _request(url, payload, headers)
    for line in _iter_lines(resp, cancel):
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("error"):
            raise ProviderError(str(parsed.get("error"))[:300])
        message = parsed.get("message") or {}
        text = message.get("content") or ""
        if text:
            yield text
        if parsed.get("done"):
            break
