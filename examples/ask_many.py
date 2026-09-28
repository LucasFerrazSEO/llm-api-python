#!/usr/bin/env python3
"""
ask_many.py - send the same prompt to every provider that has a key configured
and print each reply with the response time. Shows how to import the scripts
from your own code.

    python examples/ask_many.py "Explain recursion in one sentence"
    python examples/ask_many.py --only claude,gemini "Hello"

Part of https://github.com/LucasFerrazSEO/llm-api-python (MIT License).
"""

import argparse
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# module name -> environment variable that must be set to use it
PROVIDERS = {
    "chatgpt": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "grok": "XAI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "maritaca": "MARITACA_API_KEY",
    "ollama": "OLLAMA_MODEL",  # local: set OLLAMA_MODEL to include it
}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Send one prompt to every configured provider.")
    parser.add_argument("prompt")
    parser.add_argument("--only", help="comma-separated list, for example: claude,gemini")
    parser.add_argument("--system", help="system prompt for all providers")
    args = parser.parse_args()

    importlib.import_module("chatgpt").load_dotenv()  # every file has the same .env loader
    if os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    wanted = [name.strip() for name in args.only.split(",")] if args.only else list(PROVIDERS)
    unknown = [name for name in wanted if name not in PROVIDERS]
    if unknown:
        parser.error(f"unknown provider(s): {', '.join(unknown)}")

    ran = 0
    for name in wanted:
        if not os.environ.get(PROVIDERS[name]):
            print(f"-- {name}: skipped ({PROVIDERS[name]} is not set)\n")
            continue
        module = importlib.import_module(name)
        start = time.perf_counter()
        try:
            text, sources, _ = module.complete([{"role": "user", "content": args.prompt}], system=args.system)
        except module.APIError as exc:
            print(f"-- {name}: error\n{exc}\n")
            continue
        ran += 1
        print(f"-- {name} ({time.perf_counter() - start:.1f}s)\n{text.strip()}")
        for number, url in enumerate(sources, 1):
            print(f"  [{number}] {url}")
        print()
    if not ran:
        print("No provider answered. Copy .env.example to .env and fill in at least one key.")
    return 0 if ran else 1


if __name__ == "__main__":
    sys.exit(main())
