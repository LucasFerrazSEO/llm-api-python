#!/usr/bin/env python3
"""
gemini_sdk.py - the same send/receive as ../gemini.py, using the official
Google Gen AI SDK (package google-genai).

    pip install -r requirements.txt
    python gemini_sdk.py "Explain recursion in one sentence"
    python gemini_sdk.py --stream "Write a haiku about Python"
    python gemini_sdk.py --search "What changed in Python this month?"

The SDK reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.

Part of https://github.com/LucasFerrazSEO/llm-api-python (MIT License).
"""

import argparse
import os
import sys

from google import genai
from google.genai import errors, types
from dotenv import load_dotenv

load_dotenv()  # reads the .env file (python-dotenv also searches parent folders)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a prompt to Gemini with the official SDK.")
    parser.add_argument("prompt")
    parser.add_argument("--system", help="system prompt (instructions)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--search", action="store_true", help="ground the answer with Google Search")
    args = parser.parse_args()

    config = types.GenerateContentConfig(
        system_instruction=args.system,
        tools=[types.Tool(google_search=types.GoogleSearch())] if args.search else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    try:
        client = genai.Client()  # uses GEMINI_API_KEY or GOOGLE_API_KEY
        if not args.stream:
            response = client.models.generate_content(
                model=args.model, contents=args.prompt, config=config)
            print(response.text)
            return 0
        for chunk in client.models.generate_content_stream(
                model=args.model, contents=args.prompt, config=config):
            print(chunk.text or "", end="", flush=True)
        print()
        return 0
    except errors.APIError as exc:
        print(f"Error: HTTP {exc.code}. {exc.message}", file=sys.stderr)
    except ValueError as exc:  # raised when no API key is found
        print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
