#!/usr/bin/env python3
"""
claude_sdk.py - the same send/receive as ../claude.py, using the official
Anthropic SDK.

    pip install -r requirements.txt
    python claude_sdk.py "Explain recursion in one sentence"
    python claude_sdk.py --stream "Write a haiku about Python"

The SDK reads ANTHROPIC_API_KEY from the environment.

Part of https://github.com/LucasFerrazSEO/llm-api-python (MIT License).
"""

import argparse
import os
import sys

import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads the .env file (python-dotenv also searches parent folders)

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a prompt to Claude with the official SDK.")
    parser.add_argument("prompt")
    parser.add_argument("--system", help="system prompt (instructions)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY
    options = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": args.prompt}],
    }
    if args.system:
        options["system"] = args.system
    try:
        if not args.stream:
            message = client.messages.create(**options)
            print("".join(block.text for block in message.content if block.type == "text"))
            return 0
        with client.messages.stream(**options) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        print()
        return 0
    except anthropic.AuthenticationError:
        print("Error: invalid or missing ANTHROPIC_API_KEY.", file=sys.stderr)
    except anthropic.RateLimitError as exc:
        print(f"Error: rate limit exceeded. {exc.message}", file=sys.stderr)
    except anthropic.APIConnectionError:
        print("Error: could not reach the Anthropic API.", file=sys.stderr)
    except anthropic.APIStatusError as exc:
        print(f"Error: HTTP {exc.status_code}. {exc.message}", file=sys.stderr)
    except (anthropic.AnthropicError, TypeError) as exc:  # for example, no API key
        print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
