"""Provider catalog, settings redaction, and streamed completions."""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from model_bridge import LLMBridge
from providers import (
    BY_ID,
    CATALOG,
    ProviderError,
    ProviderSettings,
    anthropic_parts,
    endpoint_url,
    settings_from_env,
)


REQUIRED = (
    "local",
    "ollama",
    "ollama_cloud",
    "openai",
    "anthropic",
    "gemini",
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
)


class CatalogTests(unittest.TestCase):
    def test_ids_are_unique_and_complete(self):
        ids = [spec.id for spec in CATALOG]
        self.assertEqual(ids, list(REQUIRED))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(BY_ID["local"].kind, "local")
        self.assertTrue(BY_ID["ollama_cloud"].key_required)
        self.assertEqual(BY_ID["anthropic"].kind, "anthropic")
        self.assertEqual(BY_ID["ollama"].kind, "ollama")
        self.assertTrue(BY_ID["gemini"].base_url.endswith("/openai"))

    def test_endpoint_url_does_not_duplicate_a_path(self):
        self.assertEqual(
            endpoint_url("https://api.openai.com/v1", "/chat/completions"),
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(
            endpoint_url("https://api.openai.com/v1/chat/completions", "/chat/completions"),
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(
            endpoint_url("http://127.0.0.1:11434", "/api/chat"),
            "http://127.0.0.1:11434/api/chat",
        )

    def test_anthropic_merges_system_and_adjacent_roles(self):
        system, turns = anthropic_parts([
            {"role": "system", "content": "Be brief."},
            {"role": "system", "content": "Stay in English."},
            {"role": "assistant", "content": "I was here."},
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Again"},
        ])
        self.assertIn("Be brief.", system)
        self.assertIn("Stay in English.", system)
        self.assertEqual(turns[0]["role"], "user")
        self.assertEqual(turns[-1]["content"], "Hello\n\nAgain")


class SettingsTests(unittest.TestCase):
    def test_roundtrip_hides_the_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chronosoul_providers.json"
            settings = ProviderSettings(path)
            settings.update({
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test-secret-value",
            })
            public = json.dumps(settings.public())
            self.assertNotIn("sk-test-secret-value", public)
            self.assertEqual(settings.public()["provider"], "openai")
            self.assertEqual(settings.public()["model"], "gpt-4o")
            openai = next(item for item in settings.public()["providers"] if item["id"] == "openai")
            self.assertTrue(openai["has_key"])
            self.assertEqual(openai["key_source"], "saved")
            reloaded = ProviderSettings(path)
            self.assertEqual(reloaded.api_key("openai"), "sk-test-secret-value")
            self.assertTrue(oct(path.stat().st_mode & 0o777) in {"0o600", "0o644"})

    def test_clear_key_and_reject_bad_url(self):
        settings = ProviderSettings()
        settings.update({"provider": "custom", "model": "my-model", "base_url": "http://127.0.0.1:9/v1", "api_key": "secret"})
        self.assertEqual(settings.api_key(), "secret")
        settings.update({"provider": "custom", "model": "my-model", "base_url": "http://127.0.0.1:9/v1", "clear_key": True})
        self.assertEqual(settings.api_key(), "")
        with self.assertRaises(ProviderError):
            settings.update({"provider": "custom", "model": "my-model", "base_url": "javascript:alert(1)"})
        with self.assertRaises(ProviderError):
            settings.update({"provider": "not-a-provider", "model": "x"})

    def test_env_key_is_used_but_not_copied_into_the_file(self):
        names = ["CHRONOSOUL_PROVIDER", "CHRONOSOUL_MODEL", "LLM_MODEL", "OLLAMA_MODEL", "OLLAMA_HOST", "OLLAMA_ENDPOINT"]
        names += [spec.env_key for spec in CATALOG if spec.env_key]
        names.append("GOOGLE_API_KEY")
        previous = {name: os.environ.get(name) for name in names}
        for name in names:
            os.environ.pop(name, None)
        os.environ["GEMINI_API_KEY"] = "gemini-from-env"
        try:
            settings = ProviderSettings()
            self.assertEqual(settings.api_key("gemini"), "gemini-from-env")
            self.assertEqual(settings.key_source("gemini"), "env")
            self.assertNotIn("gemini", settings.data["keys"])
            picked = settings_from_env()
            self.assertEqual(picked.spec().id, "gemini")
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_missing_key_marks_the_voice_not_ready(self):
        info = {"status": "booting", "detail": "", "steering": "on"}
        settings = ProviderSettings(data={"provider": "anthropic"})
        settings.annotate(info)
        self.assertEqual(info["status"], "error")
        self.assertIn("API key", info["detail"])
        self.assertEqual(info["provider"], "anthropic")
        self.assertEqual(info["steering"], "off")

    def test_local_selection_leaves_the_loader_status_alone(self):
        info = {"status": "loading", "detail": "Waking the substrate voice"}
        ProviderSettings().annotate(info)
        self.assertEqual(info["status"], "loading")
        self.assertEqual(info["provider"], "local")


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.seen = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                outer = self.server.seen  # type: ignore[attr-defined]
                outer["path"] = self.path
                outer["auth"] = self.headers.get("Authorization")
                outer["anthropic"] = self.headers.get("anthropic-version")
                outer["body"] = body
                if body.get("model") == "missing-model":
                    raw = b'{"error":{"message":"model not found"}}'
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if self.path.endswith("/chat/completions"):
                    payload = (
                        'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
                        'data: {"choices":[{"delta":{"content":"there"}}]}\n\n'
                        "data: [DONE]\n\n"
                    )
                elif self.path.endswith("/messages"):
                    payload = (
                        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Claude"}}\n\n'
                    )
                elif self.path.endswith("/api/chat"):
                    payload = (
                        '{"message":{"content":"from "},"done":false}\n'
                        '{"message":{"content":"ollama"},"done":true}\n'
                    )
                else:
                    self.send_error(404)
                    return
                raw = payload.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.seen = self.seen  # type: ignore[attr-defined]
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    def _settings(self, provider, model, key="test-key"):
        base = f"http://127.0.0.1:{self.port}"
        if provider == "openai":
            base += "/v1"
        settings = ProviderSettings(data={
            "provider": provider,
            "models": {provider: model},
            "base_urls": {provider: base},
            "keys": {provider: key},
        })
        return settings

    def test_openai_stream_and_bridge(self):
        settings = self._settings("openai", "gpt-4o-mini")
        messages = [
            {"role": "system", "content": "Be kind."},
            {"role": "user", "content": "Hi"},
        ]
        self.assertEqual("".join(settings.stream(messages)), "Hello there")
        self.assertEqual(self.seen["path"], "/v1/chat/completions")
        self.assertEqual(self.seen["auth"], "Bearer test-key")
        self.assertTrue(self.seen["body"]["stream"])
        bridge = LLMBridge(settings)
        self.assertEqual(bridge.active_provider, "openai")
        self.assertEqual(bridge.generate_response("system", "Hi", _Affect()), "Hello there")

    def test_anthropic_and_ollama_streams(self):
        anthropic = self._settings("anthropic", "claude-sonnet-4-5")
        self.assertEqual("".join(anthropic.stream([{"role": "user", "content": "Hi"}])), "Claude")
        self.assertEqual(self.seen["anthropic"], "2023-06-01")
        ollama = self._settings("ollama", "llama3.2", key="")
        self.assertEqual("".join(ollama.stream([{"role": "user", "content": "Hi"}])), "from ollama")
        self.assertNotIn("Authorization", self.seen["auth"] or "")

    def test_cancel_closes_without_yielding(self):
        settings = self._settings("openai", "gpt-4o-mini")
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(list(settings.stream([{"role": "user", "content": "Hi"}], cancel)), [])

    def test_http_error_is_a_provider_error(self):
        settings = self._settings("openai", "missing-model")
        with self.assertRaises(ProviderError) as caught:
            settings.complete([{"role": "user", "content": "Hi"}])
        self.assertIn("404", str(caught.exception))
        self.assertIn("model not found", str(caught.exception))
        bridge = LLMBridge(settings)
        reply = bridge.generate_response("system", "Hi", _Affect())
        self.assertIn("Substrate Connection Fallback", reply)


class _Affect:
    valence = 0.2
    arousal = 0.3
    solitude = 0.1


class PageTests(unittest.TestCase):
    def test_settings_button_is_in_the_chamber(self):
        root = Path(__file__).resolve().parents[1] / "desktop" / "renderer"
        html = (root / "index.html").read_text(encoding="utf-8")
        script = (root / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="settingsBtn"', html)
        self.assertIn("Model providers", html)
        self.assertIn("/providers", script)
        self.assertIn("Use this voice", html)


if __name__ == "__main__":
    unittest.main()
