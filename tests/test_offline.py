"""Offline tests: every script talks to a local fake server, so no API key or credit is used.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import importlib.util
import io
import json
import os
import sys
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# file -> (URL override variable, key variable, expected Authorization prefix)
OPENAI_COMPATIBLE = {
    "chatgpt.py": ("OPENAI_API_URL", "OPENAI_API_KEY", "Bearer "),
    "grok.py": ("XAI_API_URL", "XAI_API_KEY", "Bearer "),
    "deepseek.py": ("DEEPSEEK_API_URL", "DEEPSEEK_API_KEY", "Bearer "),
    "mistral.py": ("MISTRAL_API_URL", "MISTRAL_API_KEY", "Bearer "),
    "groq.py": ("GROQ_API_URL", "GROQ_API_KEY", "Bearer "),
    "openrouter.py": ("OPENROUTER_API_URL", "OPENROUTER_API_KEY", "Bearer "),
    "qwen.py": ("DASHSCOPE_API_URL", "DASHSCOPE_API_KEY", "Bearer "),
    "kimi.py": ("MOONSHOT_API_URL", "MOONSHOT_API_KEY", "Bearer "),
    "maritaca.py": ("MARITACA_API_URL", "MARITACA_API_KEY", "Key "),
    "ollama.py": ("OLLAMA_API_URL", "OLLAMA_API_KEY", "Bearer "),
}


class FakeAPI(BaseHTTPRequestHandler):
    """Answers like the real APIs. The last request is kept in `requests` for assertions."""

    requests = []
    fail_next = []  # list of (status, body) returned before the normal answer

    def log_message(self, *args):
        pass

    def _send(self, status, body, content_type="application/json", headers=None):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        FakeAPI.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        if FakeAPI.fail_next:
            status, error = FakeAPI.fail_next.pop(0)
            return self._send(status, error, headers={"Retry-After": "0"})
        stream = body.get("stream") or "alt=sse" in self.path
        if self.path.startswith("/openai"):
            return self._openai(stream)
        if self.path.startswith("/anthropic"):
            return self._anthropic(stream)
        if self.path.startswith("/gemini"):
            return self._gemini(stream)
        if self.path.startswith("/perplexity"):
            return self._perplexity(stream)
        self._send(404, '{"error": {"message": "unknown path"}}')

    @staticmethod
    def _sse(events, named=False):
        lines = []
        for event in events:
            if named:
                lines.append(f"event: {event['type']}")
            lines.append("data: " + json.dumps(event))
            lines.append("")
        return "\n".join(lines) + "\n"

    def _openai(self, stream):
        if not stream:
            return self._send(200, json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "Hello from OpenAI format"}}],
                "citations": ["https://example.com/a"],
            }))
        events = [{"choices": [{"delta": {"content": piece}}]} for piece in ("Hel", "lo ", "stream")]
        events[-1]["search_results"] = [{"url": "https://example.com/b"}]
        self._send(200, self._sse(events) + "data: [DONE]\n\n", "text/event-stream")

    def _anthropic(self, stream):
        if not stream:
            return self._send(200, json.dumps({
                "type": "message", "content": [{"type": "text", "text": "Hello from Claude"}],
            }))
        events = [{"type": "message_start"}] + [
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": piece}}
            for piece in ("Hel", "lo ", "stream")
        ] + [{"type": "message_stop"}]
        self._send(200, self._sse(events, named=True), "text/event-stream")

    def _gemini(self, stream):
        grounding = {"groundingChunks": [{"web": {"uri": "https://example.com/g"}}]}
        if not stream:
            return self._send(200, json.dumps({"candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hello from Gemini"}]},
                "groundingMetadata": grounding,
            }]}))
        events = [{"candidates": [{"content": {"parts": [{"text": piece}]}}]} for piece in ("Hel", "lo ", "stream")]
        events[-1]["candidates"][0]["groundingMetadata"] = grounding
        self._send(200, self._sse(events), "text/event-stream")

    def _perplexity(self, stream):
        final = {
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [{
                    "type": "output_text", "text": "Hello from Perplexity",
                    "annotations": [{"type": "url_citation", "url": "https://example.com/p1"}],
                }]},
                {"type": "search_results", "results": [{"url": "https://example.com/p2"}]},
            ],
        }
        if not stream:
            return self._send(200, json.dumps(final))
        events = [{"type": "response.created"}] + [
            {"type": "response.output_text.delta", "delta": piece} for piece in ("Hel", "lo ", "stream")
        ] + [{"type": "response.completed", "response": final}]
        self._send(200, self._sse(events, named=True), "text/event-stream")


def load(filename):
    spec = importlib.util.spec_from_file_location(f"lap_{filename[:-3]}", os.path.join(ROOT, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.time.sleep = lambda seconds: None  # no real waiting in retry tests
    return module


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeAPI.requests.clear()
        FakeAPI.fail_next.clear()
        self._saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def last(self):
        return FakeAPI.requests[-1]


class OpenAICompatibleTests(Base):
    def test_every_compatible_file(self):
        for filename, (url_env, key_env, scheme) in OPENAI_COMPATIBLE.items():
            with self.subTest(file=filename):
                module = load(filename)
                os.environ[url_env] = self.base + "/openai/chat/completions"
                os.environ[key_env] = "test-key"

                text, sources, _ = module.complete(
                    [{"role": "user", "content": "Hi"}], system="Be brief", max_tokens=50)
                self.assertEqual(text, "Hello from OpenAI format")
                self.assertEqual(sources, ["https://example.com/a"])
                request = self.last()
                self.assertEqual(request["headers"]["authorization"], scheme + "test-key")
                self.assertEqual(request["body"]["messages"][0], {"role": "system", "content": "Be brief"})
                self.assertEqual(request["body"]["model"], module.DEFAULT_MODEL)
                self.assertEqual(request["body"][module.MAX_TOKENS_FIELD], 50)

                pieces = []
                text, sources, _ = module.complete(
                    [{"role": "user", "content": "Hi"}], stream=True, on_token=pieces.append)
                self.assertEqual(text, "Hello stream")
                self.assertEqual(pieces, ["Hel", "lo ", "stream"])
                self.assertEqual(sources, ["https://example.com/b"])

    def test_model_from_environment(self):
        module = load("mistral.py")
        os.environ["MISTRAL_API_URL"] = self.base + "/openai/chat/completions"
        os.environ["MISTRAL_API_KEY"] = "k"
        os.environ["MISTRAL_MODEL"] = "my-model"
        module.ask("Hi")
        self.assertEqual(self.last()["body"]["model"], "my-model")

    def test_retry_then_success(self):
        module = load("chatgpt.py")
        os.environ["OPENAI_API_URL"] = self.base + "/openai/chat/completions"
        os.environ["OPENAI_API_KEY"] = "k"
        FakeAPI.fail_next.extend([(429, '{"error": {"message": "slow down"}}'), (503, "busy")])
        with redirect_stderr(io.StringIO()):
            self.assertEqual(module.ask("Hi"), "Hello from OpenAI format")
        self.assertEqual(len(FakeAPI.requests), 3)

    def test_no_retry_when_out_of_credits(self):
        module = load("chatgpt.py")
        os.environ["OPENAI_API_URL"] = self.base + "/openai/chat/completions"
        os.environ["OPENAI_API_KEY"] = "k"
        FakeAPI.fail_next.append((429, '{"error": {"code": "insufficient_quota", "message": "no credits"}}'))
        with self.assertRaises(module.APIError) as ctx:
            module.ask("Hi")
        self.assertIn("no credits", str(ctx.exception))
        self.assertEqual(len(FakeAPI.requests), 1)

    def test_invalid_key_message(self):
        module = load("grok.py")
        os.environ["XAI_API_URL"] = self.base + "/openai/chat/completions"
        os.environ["XAI_API_KEY"] = "wrong"
        FakeAPI.fail_next.append((401, '{"error": "Incorrect API key"}'))
        with self.assertRaises(module.APIError) as ctx:
            module.ask("Hi")
        self.assertIn("XAI_API_KEY", str(ctx.exception))
        self.assertIn("Incorrect API key", str(ctx.exception))

    def test_missing_key(self):
        module = load("deepseek.py")
        os.environ.pop("DEEPSEEK_API_KEY", None)
        with self.assertRaises(module.APIError) as ctx:
            module.ask("Hi")
        self.assertIn("DEEPSEEK_API_KEY is not set", str(ctx.exception))

    def test_ollama_needs_no_key(self):
        module = load("ollama.py")
        os.environ["OLLAMA_API_URL"] = self.base + "/openai/chat/completions"
        os.environ.pop("OLLAMA_API_KEY", None)
        self.assertEqual(module.ask("Hi"), "Hello from OpenAI format")
        self.assertNotIn("authorization", self.last()["headers"])

    def test_openrouter_optional_headers(self):
        module = load("openrouter.py")
        os.environ["OPENROUTER_API_URL"] = self.base + "/openai/chat/completions"
        os.environ["OPENROUTER_API_KEY"] = "k"
        os.environ["OPENROUTER_SITE_URL"] = "https://example.com"
        os.environ["OPENROUTER_APP_NAME"] = "My app"
        module.ask("Hi")
        headers = self.last()["headers"]
        self.assertEqual(headers["http-referer"], "https://example.com")
        self.assertEqual(headers["x-openrouter-title"], "My app")

    def test_command_line(self):
        module = load("chatgpt.py")
        os.environ["OPENAI_API_URL"] = self.base + "/openai/chat/completions"
        os.environ["OPENAI_API_KEY"] = "k"
        out = io.StringIO()
        with redirect_stdout(out):
            code = module.main(["--stream", "Hi"])
        self.assertEqual(code, 0)
        self.assertIn("Hello stream", out.getvalue())
        self.assertIn("[1] https://example.com/b", out.getvalue())

    def test_chat_keeps_history(self):
        module = load("chatgpt.py")
        os.environ["OPENAI_API_URL"] = self.base + "/openai/chat/completions"
        os.environ["OPENAI_API_KEY"] = "k"
        answers = iter(["first", "second", "/exit"])
        module.input = lambda prompt="": next(answers)
        with redirect_stdout(io.StringIO()):
            module.chat()
        roles = [m["role"] for m in self.last()["body"]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])


class ClaudeTests(Base):
    def setUp(self):
        super().setUp()
        self.module = load("claude.py")
        os.environ["ANTHROPIC_API_URL"] = self.base + "/anthropic/v1/messages"
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

    def test_request_format(self):
        text = self.module.send(
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
             {"role": "user", "content": "Again"}], system="Be brief")
        self.assertEqual(text, "Hello from Claude")
        request = self.last()
        self.assertEqual(request["headers"]["x-api-key"], "test-key")
        self.assertEqual(request["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(request["body"]["system"], "Be brief")
        self.assertEqual(request["body"]["max_tokens"], self.module.DEFAULT_MAX_TOKENS)
        self.assertEqual(len(request["body"]["messages"]), 3)

    def test_stream(self):
        pieces = []
        text, _, _ = self.module.complete([{"role": "user", "content": "Hi"}], stream=True, on_token=pieces.append)
        self.assertEqual(text, "Hello stream")
        self.assertEqual(pieces, ["Hel", "lo ", "stream"])

    def test_overloaded_is_retried(self):
        FakeAPI.fail_next.append((529, '{"type": "error", "error": {"message": "Overloaded"}}'))
        with redirect_stderr(io.StringIO()):
            self.assertEqual(self.module.ask("Hi"), "Hello from Claude")


class GeminiTests(Base):
    def setUp(self):
        super().setUp()
        self.module = load("gemini.py")
        os.environ["GEMINI_API_URL"] = self.base + "/gemini/v1beta"
        os.environ["GEMINI_API_KEY"] = "test-key"

    def test_request_format(self):
        text, sources, _ = self.module.complete(
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"},
             {"role": "user", "content": "Again"}], system="Be brief", max_tokens=20, search=True)
        self.assertEqual(text, "Hello from Gemini")
        self.assertEqual(sources, ["https://example.com/g"])
        request = self.last()
        self.assertEqual(request["path"], f"/gemini/v1beta/models/{self.module.DEFAULT_MODEL}:generateContent")
        self.assertEqual(request["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual([c["role"] for c in request["body"]["contents"]], ["user", "model", "user"])
        self.assertEqual(request["body"]["systemInstruction"], {"parts": [{"text": "Be brief"}]})
        self.assertEqual(request["body"]["generationConfig"], {"maxOutputTokens": 20})
        self.assertEqual(request["body"]["tools"], [{"google_search": {}}])

    def test_stream(self):
        pieces = []
        text, sources, _ = self.module.complete([{"role": "user", "content": "Hi"}], stream=True,
                                                on_token=pieces.append)
        self.assertEqual(text, "Hello stream")
        self.assertEqual(pieces, ["Hel", "lo ", "stream"])
        self.assertIn("streamGenerateContent?alt=sse", self.last()["path"])

    def test_google_api_key_fallback(self):
        del os.environ["GEMINI_API_KEY"]
        os.environ["GOOGLE_API_KEY"] = "other-key"
        self.module.ask("Hi")
        self.assertEqual(self.last()["headers"]["x-goog-api-key"], "other-key")


class PerplexityTests(Base):
    def setUp(self):
        super().setUp()
        self.module = load("perplexity.py")
        os.environ["PERPLEXITY_API_URL"] = self.base + "/perplexity/v1/agent"
        os.environ["PERPLEXITY_API_KEY"] = "test-key"

    def test_request_format(self):
        text, sources, _ = self.module.complete([{"role": "user", "content": "Hi"}], system="Be brief")
        self.assertEqual(text, "Hello from Perplexity")
        self.assertEqual(sources, ["https://example.com/p1", "https://example.com/p2"])
        body = self.last()["body"]
        self.assertEqual(body["preset"], "fast")
        self.assertEqual(body["instructions"], "Be brief")
        self.assertEqual(body["input"], [{"type": "message", "role": "user", "content": "Hi"}])
        self.assertNotIn("model", body)

    def test_model_replaces_preset(self):
        self.module.complete([{"role": "user", "content": "Hi"}], model="vendor/model")
        body = self.last()["body"]
        self.assertEqual(body["model"], "vendor/model")
        self.assertNotIn("preset", body)
        self.assertEqual(body["tools"], [{"type": "web_search"}])

    def test_stream(self):
        pieces = []
        text, sources, _ = self.module.complete([{"role": "user", "content": "Hi"}], stream=True,
                                                on_token=pieces.append)
        self.assertEqual(text, "Hello stream")
        self.assertEqual(pieces, ["Hel", "lo ", "stream"])
        self.assertEqual(sources, ["https://example.com/p1", "https://example.com/p2"])


class DotenvTests(unittest.TestCase):
    def test_dotenv_does_not_override_environment(self):
        module = load("chatgpt.py")
        import tempfile
        saved_cwd, saved_env = os.getcwd(), dict(os.environ)
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, ".env"), "w", encoding="utf-8") as handle:
                handle.write('# comment\nexport LAP_A="from file"\nLAP_B=\'quoted\'\nLAP_C=file\n')
            os.environ["LAP_C"] = "from env"
            try:
                os.chdir(folder)
                module.load_dotenv()
                self.assertEqual(os.environ["LAP_A"], "from file")
                self.assertEqual(os.environ["LAP_B"], "quoted")
                self.assertEqual(os.environ["LAP_C"], "from env")
            finally:
                os.chdir(saved_cwd)
                os.environ.clear()
                os.environ.update(saved_env)


if __name__ == "__main__":
    unittest.main()
