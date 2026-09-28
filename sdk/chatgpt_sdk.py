#!/usr/bin/env python3
"""
chatgpt_sdk.py - the same send/receive as ../chatgpt.py, using the official
OpenAI SDK and the Responses API (the one OpenAI recommends for new projects).

    pip install -r requirements.txt
    python chatgpt_sdk.py "Explain recursion in one sentence"
    python chatgpt_sdk.py --stream "Write a haiku about Python"

The SDK reads OPENAI_API_KEY from the environment.

Part of https://github.com/LucasFerrazSEO/llm-api-python (MIT License).
"""

import argparse
import os
import sys

import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads the .env file (python-dotenv also searches parent folders)

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-luna")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a prompt to OpenAI with the official SDK.")
    parser.add_argument("prompt")
    parser.add_argument("--system", help="system prompt (instructions)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    try:
        client = OpenAI()  # uses OPENAI_API_KEY
        if not args.stream:
            response = client.responses.create(
                model=args.model, instructions=args.system, input=args.prompt)
            print(response.output_text)
            return 0
        stream = client.responses.create(
            model=args.model, instructions=args.system, input=args.prompt, stream=True)
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
        print()
        return 0
    except openai.AuthenticationError:
        print("Error: invalid or missing OPENAI_API_KEY.", file=sys.stderr)
    except openai.RateLimitError as exc:
        print(f"Error: rate limit or quota exceeded. {exc.message}", file=sys.stderr)
    except openai.APIConnectionError:
        print("Error: could not reach the OpenAI API.", file=sys.stderr)
    except openai.APIStatusError as exc:
        print(f"Error: HTTP {exc.status_code}. {exc.message}", file=sys.stderr)
    except openai.OpenAIError as exc:  # for example, no API key
        print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
