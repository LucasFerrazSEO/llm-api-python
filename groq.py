#!/usr/bin/env python3
"""
groq.py - send a prompt to Groq through its API and print the reply.

Standard library only: copy this single file, set GROQ_API_KEY and run it.

    python groq.py "Explain recursion in one sentence"
    python groq.py --stream "Write a haiku about Python"
    python groq.py --chat
    echo "Summarize this text: ..." | python groq.py -

Or import it from your own code:

    from groq import ask, send
    print(ask("Hello!"))
    print(send([{"role": "user", "content": "Hi"}], system="Answer in Portuguese"))

Groq is a fast inference service for open models (Llama, gpt-oss and
others), not the same thing as xAI Grok (see grok.py).

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

# ---- Provider settings (the only block that changes between the files) ----
PROVIDER = "Groq"
API_URL = "https://api.groq.com/openai/v1/chat/completions"
URL_ENV = "GROQ_API_URL"  # optional: override the endpoint (proxy, gateway, test server)
KEY_ENV = "GROQ_API_KEY"
KEY_REQUIRED = True
AUTH_SCHEME = "Bearer"  # Authorization: <scheme> <key>
KEYS_PAGE = "https://console.groq.com/keys"
MODEL_ENV = "GROQ_MODEL"
DEFAULT_MODEL = "llama-3.1-8b-instant"  # model ids change often: check https://console.groq.com/docs/models
MAX_TOKENS_FIELD = "max_tokens"
EXTRA_HEADERS_ENV = {}  # header name -> environment variable
CONNECT_HINT = ""
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
    headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT}
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        headers["Authorization"] = f"{AUTH_SCHEME} {key}"
    elif KEY_REQUIRED:
        raise APIError(
            f"{KEY_ENV} is not set. Create a key at {KEYS_PAGE} and put it "
            f"in a .env file (see .env.example) or in your environment."
        )
    for header, env_name in EXTRA_HEADERS_ENV.items():
        value = os.environ.get(env_name, "").strip()
        if value:
            headers[header] = value
    return headers


def _error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:500]
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        error = data.get("error", data)
        if isinstance(error, dict):
            return str(error.get("message") or error.get("detail") or error)[:500]
        return str(error or data.get("message") or data.get("detail") or data)[:500]
    return str(data)[:500]


def _explain(status: int, body: str) -> str:
    detail = _error_message(body)
    hints = {
        400: "The request was rejected. Check the model name and the parameters.",
        401: f"Invalid or missing API key. Check {KEY_ENV}.",
        402: "Payment required: the account has no credits or no billing set up.",
        403: "Access denied: the key has no permission for this model or endpoint.",
        404: f"Not found: check the model name ({MODEL_ENV}) or the endpoint URL.",
        413: "The request is too large for this model.",
        429: "Rate limit or quota exceeded. Wait, or check the plan and credits.",
    }
    if status == 400 and "key" in detail.lower():
        status_hint = hints[401]
    else:
        status_hint = hints.get(status, "The provider returned an error.")
    return f"HTTP {status}. {status_hint}" + (f"\nProvider message: {detail}" if detail else "")


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
            out_of_credits = "insufficient_quota" in body or "insufficient balance" in body.lower()
            if exc.code in RETRY_STATUS and not out_of_credits and attempt < MAX_RETRIES:
                wait = _retry_after(exc.headers.get("Retry-After"), attempt)
                print(f"[HTTP {exc.code}] retrying in {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise APIError(_explain(exc.code, body)) from None
        except urllib.error.URLError as exc:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise APIError(f"Could not reach {url}: {exc.reason}. {CONNECT_HINT}".strip()) from None
    raise APIError("Unreachable")  # pragma: no cover


def _retry_after(value, attempt: int) -> float:
    try:
        return min(max(float(value), 1.0), 60.0)
    except (TypeError, ValueError):
        return float(2 ** (attempt + 1))


def _text(content) -> str:
    """Message content can be a string or a list of parts, depending on the provider."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
            if not isinstance(part, dict) or part.get("type", "text") == "text"
        )
    return ""


def _sources(data: dict) -> list:
    """Web sources, for providers that search the web (Perplexity, for example)."""
    found = []
    for item in data.get("citations") or []:
        found.append(item if isinstance(item, str) else item.get("url", ""))
    for item in data.get("search_results") or []:
        if isinstance(item, dict) and item.get("url"):
            found.append(item["url"])
    return [url for url in dict.fromkeys(found) if url]


def _iter_sse(response):
    """Yield the JSON events of a Server-Sent Events stream."""
    for raw in response:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            break
        try:
            yield json.loads(chunk)
        except ValueError:
            continue


def complete(messages, model=None, system=None, stream=False, max_tokens=None,
             temperature=None, on_token=None):
    """Send the conversation and return (reply_text, sources, raw_response).

    messages: list of {"role": "user" | "assistant", "content": "..."}
    on_token: optional function called with each piece of text while streaming.
    raw_response is the full JSON (or the list of stream events when stream=True).
    """
    chat_messages = ([{"role": "system", "content": system}] if system else []) + list(messages)
    payload = {
        "model": model or os.environ.get(MODEL_ENV, "").strip() or DEFAULT_MODEL,
        "messages": chat_messages,
    }
    if max_tokens:
        payload[MAX_TOKENS_FIELD] = int(max_tokens)
    if temperature is not None:
        payload["temperature"] = float(temperature)

    if not stream:
        with _post(payload) as response:
            data = json.loads(response.read().decode("utf-8"))
        if data.get("error"):
            raise APIError(f"Provider message: {_error_message(json.dumps(data))}")
        choices = data.get("choices") or [{}]
        text = _text((choices[0].get("message") or {}).get("content"))
        return text, _sources(data), data

    payload["stream"] = True
    parts, sources, events = [], [], []
    with _post(payload) as response:
        for event in _iter_sse(response):
            events.append(event)
            if event.get("error"):
                raise APIError(f"Provider message: {_error_message(json.dumps(event))}")
            for url in _sources(event):
                if url not in sources:
                    sources.append(url)
            choices = event.get("choices") or [{}]
            piece = _text((choices[0].get("delta") or {}).get("content"))
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
    parser.add_argument("--system", help="system prompt (instructions for the model)")
    parser.add_argument("--model", help=f"model id (default: ${MODEL_ENV} or {DEFAULT_MODEL})")
    parser.add_argument("--max-tokens", type=int, help="maximum length of the reply, in tokens")
    parser.add_argument("--temperature", type=float, help="sampling temperature (not every model accepts it)")
    parser.add_argument("--json", action="store_true", help="print the raw JSON response")
    args = parser.parse_args(argv)

    load_dotenv()
    options = {"model": args.model, "system": args.system, "max_tokens": args.max_tokens,
               "temperature": args.temperature}
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
