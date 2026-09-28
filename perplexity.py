#!/usr/bin/env python3
"""
perplexity.py - send a prompt to Perplexity through the Agent API and print the
reply with its web sources.

Standard library only: copy this single file, set PERPLEXITY_API_KEY and run it.

    python perplexity.py "What changed in Python this month?"
    python perplexity.py --stream "Latest news about the James Webb telescope"
    python perplexity.py --preset high "Compare the main vector databases"
    python perplexity.py --chat
    echo "Summarize this text: ..." | python perplexity.py -

Or import it from your own code:

    from perplexity import ask, complete
    print(ask("Hello!"))
    text, sources, raw = complete([{"role": "user", "content": "Hi"}])

Perplexity ended support for the old Sonar Chat Completions endpoint on
2026-09-27. This file uses the Agent API (POST /v1/agent): the conversation
goes in "input", the system prompt in "instructions", and you choose either a
preset (fast, low, medium, high, xhigh) or a "provider/model" id.

Part of https://github.com/LucasFerrazSEO/llm-api-python (MIT License).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ---- Provider settings ----------------------------------------------------
PROVIDER = "Perplexity"
API_URL = "https://api.perplexity.ai/v1/agent"
URL_ENV = "PERPLEXITY_API_URL"  # optional: override the endpoint (proxy, gateway, test server)
KEY_ENV = "PERPLEXITY_API_KEY"
KEYS_PAGE = "https://www.perplexity.ai/account/api"
PRESET_ENV = "PERPLEXITY_PRESET"
DEFAULT_PRESET = "fast"  # fast searches the web by default; see https://docs.perplexity.ai
MODEL_ENV = "PERPLEXITY_MODEL"  # optional "provider/model" id; when set, it replaces the preset
DEFAULT_MAX_TOKENS = 4096  # some models behind the Agent API require max_output_tokens
# ---------------------------------------------------------------------------

USER_AGENT = "llm-api-python/1.0 (+https://github.com/LucasFerrazSEO/llm-api-python)"
TIMEOUT = 300  # seconds (web research can take a while)
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
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise APIError(
            f"{KEY_ENV} is not set. Create a key at {KEYS_PAGE} and put it "
            f"in a .env file (see .env.example) or in your environment."
        )
    return {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT,
            "Authorization": f"Bearer {key}"}


def _error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:500]
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        return str(error.get("message") or error)[:500]
    return str(error)[:500]


def _explain(status: int, body: str) -> str:
    detail = _error_message(body)
    hints = {
        400: "The request was rejected. Check the preset or model and the parameters.",
        401: f"Invalid or missing API key. Check {KEY_ENV}.",
        402: "Payment required: the account has no credits or no billing set up.",
        403: "Access denied: the key has no permission for this model or endpoint.",
        404: "Not found: check the endpoint URL and the model id.",
        429: "Rate limit or quota exceeded. Wait, or check the plan and credits.",
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


def _post(payload: dict):
    """POST the payload and return the open HTTP response, retrying when it makes sense."""
    url = os.environ.get(URL_ENV, "").strip() or API_URL
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
    """Yield (event_name, data) pairs of a Server-Sent Events stream."""
    name = ""
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("event:"):
            name = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            data = json.loads(chunk)
        except ValueError:
            continue
        yield (data.get("type") or name) if isinstance(data, dict) else name, data
        name = ""


def _output_text(data: dict) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts = []
    for item in data.get("output") or []:
        if item.get("type") == "message":
            for block in item.get("content") or []:
                if block.get("type") == "output_text":
                    parts.append(block.get("text", ""))
    return "".join(parts)


def _sources(data) -> list:
    """Collect every URL cited in the answer or returned by the web search."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") in ("url_citation", "search_result") and node.get("url"):
                found.append(node["url"])
            if node.get("type") in ("search_results", "response.reasoning.search_results"):
                for result in node.get("results") or []:
                    if isinstance(result, dict) and result.get("url"):
                        found.append(result["url"])
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return [url for url in dict.fromkeys(found) if url]


def complete(messages, model=None, system=None, stream=False, max_tokens=None,
             temperature=None, on_token=None, preset=None):
    """Send the conversation and return (reply_text, sources, raw_response).

    messages: list of {"role": "user" | "assistant", "content": "..."}
    model: optional "provider/model" id; without it the preset is used.
    preset: fast, low, medium, high or xhigh (default: $PERPLEXITY_PRESET or fast).
    on_token: optional function called with each piece of text while streaming.
    raw_response is the full JSON (or the list of stream events when stream=True).
    """
    payload = {
        "input": [{"type": "message", "role": m["role"], "content": m["content"]} for m in messages],
        "max_output_tokens": int(max_tokens or DEFAULT_MAX_TOKENS),
    }
    model = model or os.environ.get(MODEL_ENV, "").strip()
    if model:
        payload["model"] = model
        payload["tools"] = [{"type": "web_search"}]
    else:
        payload["preset"] = preset or os.environ.get(PRESET_ENV, "").strip() or DEFAULT_PRESET
    if system:
        payload["instructions"] = system
    if temperature is not None:
        payload["temperature"] = float(temperature)

    if not stream:
        with _post(payload) as response:
            data = json.loads(response.read().decode("utf-8"))
        if data.get("error") or data.get("status") == "failed":
            raise APIError(f"Provider message: {_error_message(json.dumps(data))}")
        return _output_text(data), _sources(data), data

    payload["stream"] = True
    parts, sources, events = [], [], []
    with _post(payload) as response:
        for kind, event in _iter_sse(response):
            events.append(event)
            if kind in ("error", "response.failed"):
                raise APIError(f"Provider message: {_error_message(json.dumps(event))}")
            for url in _sources(event):
                if url not in sources:
                    sources.append(url)
            if kind == "response.output_text.delta":
                piece = event.get("delta", "")
                if isinstance(piece, str) and piece:
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
    label = (options.get("model") or os.environ.get(MODEL_ENV, "").strip()
             or f"preset {options.get('preset') or os.environ.get(PRESET_ENV, '').strip() or DEFAULT_PRESET}")
    print(f"{PROVIDER} | {label} | /reset clears the history, /exit quits")
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
    parser.add_argument("--system", help="system prompt (instructions for the model)")
    parser.add_argument("--preset", choices=["fast", "low", "medium", "high", "xhigh"],
                        help=f"research depth (default: ${PRESET_ENV} or {DEFAULT_PRESET})")
    parser.add_argument("--model", help='"provider/model" id; replaces the preset')
    parser.add_argument("--max-tokens", type=int, help=f"maximum length of the reply (default {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, help="sampling temperature (not every model accepts it)")
    parser.add_argument("--json", action="store_true", help="print the raw JSON response")
    args = parser.parse_args(argv)

    load_dotenv()
    options = {"model": args.model, "system": args.system, "max_tokens": args.max_tokens,
               "temperature": args.temperature, "preset": args.preset}
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
