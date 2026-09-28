#!/usr/bin/env python3
"""
gemini.py - send a prompt to Google Gemini through the Gemini API and print the reply.

Standard library only: copy this single file, set GEMINI_API_KEY and run it.

    python gemini.py "Explain recursion in one sentence"
    python gemini.py --stream "Write a haiku about Python"
    python gemini.py --search "What changed in Python this month?"
    python gemini.py --chat
    echo "Summarize this text: ..." | python gemini.py -

Or import it from your own code:

    from gemini import ask, send
    print(ask("Hello!"))
    print(send([{"role": "user", "content": "Hi"}], system="Answer in Portuguese"))

Gemini uses its own format (not the OpenAI one): the assistant role is called
"model", text goes in "parts" and the system prompt is "systemInstruction".
This file converts the usual {"role", "content"} messages for you.

Part of https://github.com/LucasFerrazSEO/llm-api-python (MIT License).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# ---- Provider settings ----------------------------------------------------
PROVIDER = "Google Gemini"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
URL_ENV = "GEMINI_API_URL"  # optional: override the base URL (proxy, gateway, test server)
KEY_ENVS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")  # the first one set is used
KEYS_PAGE = "https://aistudio.google.com/apikey"
MODEL_ENV = "GEMINI_MODEL"
DEFAULT_MODEL = "gemini-3.5-flash-lite"  # model ids change often: check https://ai.google.dev/gemini-api/docs/models
# ---------------------------------------------------------------------------

USER_AGENT = "llm-api-python/1.0 (+https://github.com/LucasFerrazSEO/llm-api-python)"
TIMEOUT = 180  # seconds
MAX_RETRIES = 3
RETRY_STATUS = (429, 500, 502, 503, 504)


class APIError(Exception):
    """Raised with a readable message when the API call fails."""


def load_dotenv(filename: str = ".env") -> None:
    """Load KEY=VALUE lines from .env (current folder, then this file's folder).

    Variables already set in the environment are never overwritten.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in dict.fromkeys((os.getcwd(), here)):
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8-sig") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name = name.strip()
                if name.startswith("export "):
                    name = name[len("export "):].strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                os.environ.setdefault(name, value)


def _headers() -> dict:
    key = next((os.environ[name].strip() for name in KEY_ENVS if os.environ.get(name, "").strip()), "")
    if not key:
        raise APIError(
            f"{KEY_ENVS[0]} is not set. Create a key at {KEYS_PAGE} and put it "
            f"in a .env file (see .env.example) or in your environment."
        )
    return {"Content-Type": "application/json", "Accept": "application/json",
            "User-Agent": USER_AGENT, "x-goog-api-key": key}


def _error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:500]
    if isinstance(data, list) and data:
        data = data[0]
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        return str(error.get("message") or error)[:500]
    return str(error)[:500]


def _explain(status: int, body: str) -> str:
    detail = _error_message(body)
    hints = {
        400: "The request was rejected. Check the model name, the parameters and the key.",
        401: f"Invalid or missing API key. Check {KEY_ENVS[0]}.",
        403: "Access denied: the key has no permission for this model, or the API is not enabled.",
        404: f"Not found: check the model name ({MODEL_ENV}) or the endpoint URL.",
        429: "Rate limit or quota exceeded. Wait, or check the plan and billing.",
    }
    if status == 400 and "key" in detail.lower():
        status_hint = hints[401]
    else:
        status_hint = hints.get(status, "The provider returned an error.")
    return f"HTTP {status}. {status_hint}" + (f"\nProvider message: {detail}" if detail else "")


def _retry_after(value, attempt: int) -> float:
    try:
        return min(max(float(value), 1.0), 60.0)
    except (TypeError, ValueError):
        return float(2 ** (attempt + 1))


def _post(url: str, payload: dict):
    """POST the payload and return the open HTTP response, retrying when it makes sense."""
    data = json.dumps(payload).encode("utf-8")
    headers = _headers()
    for attempt in range(MAX_RETRIES + 1):
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            return urllib.request.urlopen(request, timeout=TIMEOUT)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code in RETRY_STATUS and attempt < MAX_RETRIES:
                wait = _retry_after(exc.headers.get("Retry-After"), attempt)
                print(f"[HTTP {exc.code}] retrying in {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise APIError(_explain(exc.code, body)) from None
        except urllib.error.URLError as exc:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise APIError(f"Could not reach {url}: {exc.reason}") from None
    raise APIError("Unreachable")  # pragma: no cover


def _iter_sse(response):
    """Yield the JSON events of a Server-Sent Events stream."""
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        try:
            yield json.loads(line[5:].strip())
        except ValueError:
            continue


def _check(data: dict) -> None:
    if data.get("error"):
        raise APIError(f"Provider message: {_error_message(json.dumps(data))}")
    reason = (data.get("promptFeedback") or {}).get("blockReason")
    if reason:
        raise APIError(f"The prompt was blocked by Gemini (reason: {reason}).")


def _chunk_text(data: dict) -> str:
    candidates = data.get("candidates") or [{}]
    parts = (candidates[0].get("content") or {}).get("parts") or []
    # Parts marked "thought" are the model's reasoning summary, not the answer.
    return "".join(part.get("text", "") for part in parts if not part.get("thought"))


def _sources(data: dict) -> list:
    """Web sources used when Google Search grounding (--search) is on."""
    candidates = data.get("candidates") or [{}]
    chunks = (candidates[0].get("groundingMetadata") or {}).get("groundingChunks") or []
    urls = [(chunk.get("web") or {}).get("uri", "") for chunk in chunks]
    return [url for url in dict.fromkeys(urls) if url]


def complete(messages, model=None, system=None, stream=False, max_tokens=None,
             temperature=None, on_token=None, search=False):
    """Send the conversation and return (reply_text, sources, raw_response).

    messages: list of {"role": "user" | "assistant", "content": "..."}
    on_token: optional function called with each piece of text while streaming.
    search: True turns on grounding with Google Search (sources come back as URLs).
    raw_response is the full JSON (or the list of stream events when stream=True).
    """
    model = model or os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
    payload = {
        "contents": [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ]
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    config = {}
    if max_tokens:
        config["maxOutputTokens"] = int(max_tokens)
    if temperature is not None:
        config["temperature"] = float(temperature)
    if config:
        payload["generationConfig"] = config
    if search:
        payload["tools"] = [{"google_search": {}}]

    base = (os.environ.get(URL_ENV, "").strip() or API_BASE).rstrip("/")
    endpoint = f"{base}/models/{urllib.parse.quote(model, safe='-._')}"

    if not stream:
        with _post(f"{endpoint}:generateContent", payload) as response:
            data = json.loads(response.read().decode("utf-8"))
        _check(data)
        return _chunk_text(data), _sources(data), data

    parts, sources, events = [], [], []
    with _post(f"{endpoint}:streamGenerateContent?alt=sse", payload) as response:
        for event in _iter_sse(response):
            events.append(event)
            _check(event)
            for url in _sources(event):
                if url not in sources:
                    sources.append(url)
            piece = _chunk_text(event)
            if piece:
                parts.append(piece)
                if on_token:
                    on_token(piece)
    return "".join(parts), sources, events


def send(messages, **options) -> str:
    """Send a list of messages and return only the reply text."""
    return complete(messages, **options)[0]


def ask(prompt: str, **options) -> str:
    """Send a single prompt and return the reply text."""
    return send([{"role": "user", "content": prompt}], **options)


def _print_token(piece: str) -> None:
    sys.stdout.write(piece)
    sys.stdout.flush()


def _print_sources(sources: list) -> None:
    if sources:
        print("\nSources:")
        for number, url in enumerate(sources, 1):
            print(f"  [{number}] {url}")


def chat(**options) -> None:
    """Interactive conversation in the terminal, keeping the history."""
    model = options.get("model") or os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL
    print(f"{PROVIDER} | model {model} | /reset clears the history, /exit quits")
    history = []
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return
        if user == "/reset":
            history.clear()
            print("(history cleared)")
            continue
        history.append({"role": "user", "content": user})
        print("ai> ", end="", flush=True)
        try:
            text, sources, _ = complete(
                history, on_token=_print_token if options.get("stream") else None, **options
            )
        except APIError as exc:
            history.pop()
            print(f"\nError: {exc}", file=sys.stderr)
            continue
        print() if options.get("stream") else print(text)
        _print_sources(sources)
        history.append({"role": "assistant", "content": text})


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=f"Send a prompt to {PROVIDER} and print the reply.")
    parser.add_argument("prompt", nargs="?", help='the prompt; use "-" to read it from stdin')
    parser.add_argument("--chat", action="store_true", help="interactive conversation with history")
    parser.add_argument("--stream", action="store_true", help="print the reply while it is generated")
    parser.add_argument("--search", action="store_true", help="ground the answer with Google Search")
    parser.add_argument("--system", help="system prompt (instructions for the model)")
    parser.add_argument("--model", help=f"model id (default: ${MODEL_ENV} or {DEFAULT_MODEL})")
    parser.add_argument("--max-tokens", type=int, help="maximum length of the reply, in tokens")
    parser.add_argument("--temperature", type=float, help="sampling temperature")
    parser.add_argument("--json", action="store_true", help="print the raw JSON response")
    args = parser.parse_args(argv)

    load_dotenv()
    options = {"model": args.model, "system": args.system, "max_tokens": args.max_tokens,
               "temperature": args.temperature, "search": args.search}
    try:
        if args.chat:
            chat(stream=args.stream, **options)
            return 0
        prompt = args.prompt
        if prompt == "-" or (prompt is None and not sys.stdin.isatty()):
            prompt = sys.stdin.read()
        if not prompt or not prompt.strip():
            parser.print_help()
            return 2
        stream = args.stream and not args.json
        text, sources, raw = complete([{"role": "user", "content": prompt}],
                                      stream=stream, on_token=_print_token if stream else None, **options)
        if args.json:
            print(json.dumps(raw, ensure_ascii=False, indent=2))
            return 0
        print() if stream else print(text)
        _print_sources(sources)
        return 0
    except APIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
