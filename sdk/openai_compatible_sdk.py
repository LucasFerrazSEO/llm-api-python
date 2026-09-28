#!/usr/bin/env python3
"""
openai_compatible_sdk.py - one script, many providers: the official OpenAI SDK
talks to any API that copies the Chat Completions format, only the base URL,
the key and the model change.

    pip install -r requirements.txt
    python openai_compatible_sdk.py grok "Explain recursion in one sentence"
    python openai_compatible_sdk.py deepseek --stream "Write a haiku about Python"
    python openai_compatible_sdk.py ollama "Hello"

Providers: grok, deepseek, mistral, groq, openrouter, qwen, kimi, maritaca, ollama.
Each one reads its own key variable (see PROVIDERS below and ../.env.example).

Part of https://github.com/LucasFerrazSEO/llm-api-python (MIT License).
"""

import argparse
import os
import sys

import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads the .env file (python-dotenv also searches parent folders)

# name: (base URL, key variable, model variable, default model)
PROVIDERS = {
    "grok": ("https://api.x.ai/v1", "XAI_API_KEY", "XAI_MODEL", "grok-4.3"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-flash"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", "MISTRAL_MODEL", "mistral-small-latest"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL", "llama-3.1-8b-instant"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "openrouter/auto"),
    "qwen": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "QWEN_MODEL",
             "qwen-flash"),
    "kimi": ("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", "KIMI_MODEL", "kimi-k2.6"),
    "maritaca": ("https://chat.maritaca.ai/api", "MARITACA_API_KEY", "MARITACA_MODEL", "sabiazinho-4"),
    "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY", "OLLAMA_MODEL", "llama3.2"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a prompt to an OpenAI-compatible API with the OpenAI SDK.")
    parser.add_argument("provider", choices=sorted(PROVIDERS))
    parser.add_argument("prompt")
    parser.add_argument("--system", help="system prompt (instructions)")
    parser.add_argument("--model", help="model id (default: the provider's model variable or a sensible default)")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    base_url, key_env, model_env, default_model = PROVIDERS[args.provider]
    # Ollama ignores the key locally, but the SDK requires some value.
    api_key = os.environ.get(key_env) or ("ollama" if args.provider == "ollama" else None)
    if not api_key:
        print(f"Error: {key_env} is not set.", file=sys.stderr)
        return 1
    client = OpenAI(base_url=base_url, api_key=api_key)
    model = args.model or os.environ.get(model_env) or default_model
    messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
        {"role": "user", "content": args.prompt}]
    try:
        if not args.stream:
            completion = client.chat.completions.create(model=model, messages=messages)
            print(completion.choices[0].message.content)
            return 0
        stream = client.chat.completions.create(model=model, messages=messages, stream=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
        return 0
    except openai.AuthenticationError:
        print(f"Error: invalid API key. Check {key_env}.", file=sys.stderr)
    except openai.RateLimitError as exc:
        print(f"Error: rate limit or quota exceeded. {exc.message}", file=sys.stderr)
    except openai.APIConnectionError:
        print(f"Error: could not reach {base_url}.", file=sys.stderr)
    except openai.APIStatusError as exc:
        print(f"Error: HTTP {exc.status_code}. {exc.message}", file=sys.stderr)
    except openai.OpenAIError as exc:  # for example, no API key
        print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
