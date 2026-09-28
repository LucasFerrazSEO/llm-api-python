**English** · [Português (Brasil)](README.pt-BR.md)

# llm-api-python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg) [![tests](https://github.com/LucasFerrazSEO/llm-api-python/actions/workflows/tests.yml/badge.svg)](https://github.com/LucasFerrazSEO/llm-api-python/actions/workflows/tests.yml)

`llm-api-python` is a set of ready-to-use Python scripts that send a prompt
to an AI model through its API and receive the reply: ChatGPT (OpenAI),
Claude, Gemini, Grok, Perplexity, DeepSeek, Mistral, Groq, OpenRouter,
Qwen, Kimi, Maritaca (Sabiá) and Ollama. Each script is a single file that
uses only the Python standard library, so you can copy one file, set the
API key and run it, with nothing to install.

## Contents

- [Providers](#providers)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Command line](#command-line)
- [Use it in your own code](#use-it-in-your-own-code)
- [Official SDK versions](#official-sdk-versions)
- [Send one prompt to several AIs](#send-one-prompt-to-several-ais)
- [Errors](#errors)
- [Tests](#tests)
- [FAQ](#faq)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

## Providers

| File | Provider | Key variable | Default model | API format |
|---|---|---|---|---|
| [`chatgpt.py`](chatgpt.py) | OpenAI | `OPENAI_API_KEY` | `gpt-6-luna` | Chat Completions |
| [`claude.py`](claude.py) | Anthropic Claude | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` | Messages API (own format) |
| [`gemini.py`](gemini.py) | Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-3.5-flash-lite` | generateContent (own format) |
| [`grok.py`](grok.py) | xAI Grok | `XAI_API_KEY` | `grok-4.3` | OpenAI-compatible |
| [`perplexity.py`](perplexity.py) | Perplexity | `PERPLEXITY_API_KEY` | preset `fast` | Agent API (own format) |
| [`deepseek.py`](deepseek.py) | DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-flash` | OpenAI-compatible |
| [`mistral.py`](mistral.py) | Mistral AI | `MISTRAL_API_KEY` | `mistral-small-latest` | OpenAI-compatible |
| [`groq.py`](groq.py) | Groq (fast inference of open models) | `GROQ_API_KEY` | `llama-3.1-8b-instant` | OpenAI-compatible |
| [`openrouter.py`](openrouter.py) | OpenRouter (many models, one key) | `OPENROUTER_API_KEY` | `openrouter/auto` | OpenAI-compatible |
| [`qwen.py`](qwen.py) | Alibaba Qwen (Model Studio) | `DASHSCOPE_API_KEY` | `qwen-flash` | OpenAI-compatible |
| [`kimi.py`](kimi.py) | Moonshot AI Kimi | `MOONSHOT_API_KEY` | `kimi-k2.6` | OpenAI-compatible |
| [`maritaca.py`](maritaca.py) | Maritaca AI (Sabiá, focused on Portuguese) | `MARITACA_API_KEY` | `sabiazinho-4` | OpenAI-compatible |
| [`ollama.py`](ollama.py) | Ollama (models on your own computer) | none | `llama3.2` | OpenAI-compatible |

The default models were checked against each provider's documentation on
2026-09-28. Model ids change often; to use another one, pass `--model` or
set the model variable (for example `OPENAI_MODEL`) in `.env`.

Ten of the thirteen files share the same code and differ only in a short
"Provider settings" block at the top, because those APIs copy the OpenAI
Chat Completions format. Claude, Gemini and Perplexity have their own
formats, and their files convert the usual `{"role", "content"}` messages
for you.

## Requirements

Python 3.9 or newer. The scripts in the repository root use only the
standard library. The optional [`sdk/`](sdk/) folder uses the official
packages listed in `sdk/requirements.txt`.

You need an API key from the provider you want to use (the links are in
[`.env.example`](.env.example)). API usage is billed by the provider,
separately from any ChatGPT, Claude or Gemini chat subscription. Ollama
runs locally and needs no key.

## Quick start

```bash
git clone https://github.com/LucasFerrazSEO/llm-api-python.git
cd llm-api-python
cp .env.example .env        # on Windows: copy .env.example .env
```

Open `.env`, paste your key on the right line (for example
`OPENAI_API_KEY=sk-...`) and run:

```bash
python chatgpt.py "Explain recursion in one sentence"
```

The scripts read the key from the environment or from a `.env` file in the
current folder or in the script's folder. A variable already set in the
environment always wins over the file.

## Command line

Every root script accepts the same options:

```bash
python claude.py "Your prompt"                        # send and print the reply
python claude.py --stream "Your prompt"               # print the reply while it is generated
python claude.py --chat                               # conversation with history (/reset, /exit)
python claude.py --system "Answer in Portuguese" "Hi" # system prompt
python claude.py --model claude-sonnet-5 "Hi"         # another model
python claude.py --max-tokens 300 --temperature 0.2 "Hi"
python claude.py --json "Hi"                          # print the raw JSON response
cat notes.txt | python claude.py -                    # read the prompt from stdin
```

Extra options: `gemini.py --search` grounds the answer with Google Search,
and `perplexity.py --preset fast|low|medium|high|xhigh` chooses the depth
of the web research. When the provider returns web sources (Perplexity,
Gemini with `--search`), they are printed as a numbered list after the
reply.

## Use it in your own code

Each file is also a module with three functions:

```python
from chatgpt import ask, send, complete

# one prompt, reply as text
print(ask("Write a slogan for a bakery"))

# a conversation, with a system prompt
history = [
    {"role": "user", "content": "My name is Ana."},
    {"role": "assistant", "content": "Nice to meet you, Ana."},
    {"role": "user", "content": "What is my name?"},
]
print(send(history, system="Answer briefly"))

# streaming, plus the sources and the raw response
text, sources, raw = complete(history, stream=True, on_token=lambda t: print(t, end=""))
```

The same calls work with `claude`, `gemini`, `grok` and the other files,
so switching providers means changing the import. Failures raise
`APIError` with a readable message.

## Official SDK versions

If you prefer the official libraries, the [`sdk/`](sdk/) folder has the
same send and receive with them:

```bash
pip install -r sdk/requirements.txt
python sdk/chatgpt_sdk.py "Hi"                     # OpenAI SDK, Responses API
python sdk/claude_sdk.py --stream "Hi"             # Anthropic SDK
python sdk/gemini_sdk.py --search "Hi"             # Google Gen AI SDK
python sdk/openai_compatible_sdk.py deepseek "Hi"  # OpenAI SDK pointed at another provider
```

`openai_compatible_sdk.py` shows that the OpenAI SDK works with Grok,
DeepSeek, Mistral, Groq, OpenRouter, Qwen, Kimi, Maritaca and Ollama by
changing only `base_url`, the key and the model.

## Send one prompt to several AIs

[`examples/ask_many.py`](examples/ask_many.py) sends the same prompt to
every provider with a key in `.env` and prints each reply with its
response time:

```bash
python examples/ask_many.py "Explain recursion in one sentence"
python examples/ask_many.py --only claude,gemini,grok "Hello"
```

## Errors

The scripts turn HTTP errors into short messages: invalid or missing key
(401, and the 400 that Gemini and Grok return for a bad key), no credits
(402, or OpenAI's `insufficient_quota`), no permission (403), wrong model
or URL (404), and rate limit (429). Rate limits, server errors and
Anthropic's 529 (overloaded) are retried up to three times, respecting
`Retry-After`. Running out of credits is not retried. The provider's own
message is always shown below the summary.

Each file also accepts a URL override variable (`OPENAI_API_URL`,
`ANTHROPIC_API_URL`, `GEMINI_API_URL` and so on), useful for a proxy, a
gateway, another region (Qwen) or Ollama's cloud.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests start a local fake server that answers in the OpenAI, Claude,
Gemini and Perplexity formats, so they need no key and spend no credit.
They cover the request format of every file, streaming, sources, the
system prompt, retries, error messages, the `.env` loader and the chat
history. GitHub Actions runs them on Python 3.9 and 3.13.

## FAQ

**Is the API the same as the ChatGPT, Claude or Gemini app?**
No. The API gives you the model; the apps add their own features, memory
and settings on top. Web search, for example, is on by default in
Perplexity, optional in Gemini (`--search`) and off in the other scripts
here, so answers about recent events can differ from what the app shows.

**Do I need to install anything?**
Not for the scripts in the repository root: Python 3.9 or newer is
enough. Only the `sdk/` folder needs `pip install`.

**Can I copy only one file into my project?**
Yes. Each root file is self-contained: `claude.py` does not depend on any
other file of this repository. The only shared file is `.env`, and you can
set the variables in the environment instead.

**Which provider is free?**
Ollama runs models on your computer at no cost. Some providers offer free
tiers or trial credits, and those offers change; check each provider's
pricing page.

**What is the difference between Grok and Groq?**
Grok is xAI's model (`grok.py`). Groq is a company that serves open
models such as Llama very fast (`groq.py`).

## Limitations

The default model ids and endpoints were checked on 2026-09-28 and will
age; the model variable in `.env` lets you update them without editing the
code. The scripts send and receive text only: images, audio, file uploads
and tool calling are out of scope. `perplexity.py` uses the Agent API
because Perplexity ended support for the Sonar Chat Completions endpoint
on 2026-09-27. The files `groq.py`, `mistral.py` and `ollama.py` have the
same name as those companies' pip packages, so if you install the official
package in the same folder, rename the file before importing it. The
offline tests prove the request and response handling against a fake
server that follows each documented format; they do not replace a real
call with your own key.

## Contributing

Bug reports and new providers are welcome through [GitHub Issues](https://github.com/LucasFerrazSEO/llm-api-python/issues).
A new OpenAI-compatible provider needs only a copy of `chatgpt.py` with a
new "Provider settings" block and a line in the tests.

## Author

[Lucas Ferraz](https://lucasferraz.com) is an SEO, website development and Generative Engine Optimization specialist and the founder of [Lucas Ferraz SEO](https://lucasferrazseo.com).

## License

MIT. See [LICENSE](LICENSE).
