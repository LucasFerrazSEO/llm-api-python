[English](README.md) · **Português (Brasil)**

# llm-api-python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) ![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg) [![tests](https://github.com/LucasFerrazSEO/llm-api-python/actions/workflows/tests.yml/badge.svg)](https://github.com/LucasFerrazSEO/llm-api-python/actions/workflows/tests.yml)

`llm-api-python` é um conjunto de scripts Python prontos para enviar um
prompt a um modelo de IA pela API e receber a resposta: ChatGPT (OpenAI),
Claude, Gemini, Grok, Perplexity, DeepSeek, Mistral, Groq, OpenRouter,
Qwen, Kimi, Maritaca (Sabiá) e Ollama. Cada script é um arquivo único que
usa só a biblioteca padrão do Python: basta copiar o arquivo, configurar a
chave da API e rodar, sem instalar nada.

## Conteúdo

- [Provedores](#provedores)
- [Requisitos](#requisitos)
- [Começo rápido](#começo-rápido)
- [Linha de comando](#linha-de-comando)
- [Uso no seu código](#uso-no-seu-código)
- [Versões com o SDK oficial](#versões-com-o-sdk-oficial)
- [O mesmo prompt em várias IAs](#o-mesmo-prompt-em-várias-ias)
- [Erros](#erros)
- [Testes](#testes)
- [Perguntas frequentes](#perguntas-frequentes)
- [Limitações](#limitações)
- [Contribuir](#contribuir)
- [Autor](#autor)
- [Licença](#licença)

## Provedores

| Arquivo | Provedor | Variável da chave | Modelo padrão | Formato da API |
|---|---|---|---|---|
| [`chatgpt.py`](chatgpt.py) | OpenAI | `OPENAI_API_KEY` | `gpt-6-luna` | Chat Completions |
| [`claude.py`](claude.py) | Anthropic Claude | `ANTHROPIC_API_KEY` | `claude-haiku-4-5` | Messages API (formato próprio) |
| [`gemini.py`](gemini.py) | Google Gemini | `GEMINI_API_KEY` ou `GOOGLE_API_KEY` | `gemini-3.5-flash-lite` | generateContent (formato próprio) |
| [`grok.py`](grok.py) | xAI Grok | `XAI_API_KEY` | `grok-4.3` | compatível com OpenAI |
| [`perplexity.py`](perplexity.py) | Perplexity | `PERPLEXITY_API_KEY` | preset `fast` | Agent API (formato próprio) |
| [`deepseek.py`](deepseek.py) | DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-flash` | compatível com OpenAI |
| [`mistral.py`](mistral.py) | Mistral AI | `MISTRAL_API_KEY` | `mistral-small-latest` | compatível com OpenAI |
| [`groq.py`](groq.py) | Groq (inferência rápida de modelos abertos) | `GROQ_API_KEY` | `llama-3.1-8b-instant` | compatível com OpenAI |
| [`openrouter.py`](openrouter.py) | OpenRouter (vários modelos, uma chave) | `OPENROUTER_API_KEY` | `openrouter/auto` | compatível com OpenAI |
| [`qwen.py`](qwen.py) | Alibaba Qwen (Model Studio) | `DASHSCOPE_API_KEY` | `qwen-flash` | compatível com OpenAI |
| [`kimi.py`](kimi.py) | Moonshot AI Kimi | `MOONSHOT_API_KEY` | `kimi-k2.6` | compatível com OpenAI |
| [`maritaca.py`](maritaca.py) | Maritaca AI (Sabiá, com foco em português) | `MARITACA_API_KEY` | `sabiazinho-4` | compatível com OpenAI |
| [`ollama.py`](ollama.py) | Ollama (modelos no seu próprio computador) | nenhuma | `llama3.2` | compatível com OpenAI |

Os modelos padrão foram conferidos na documentação de cada provedor em
28/09/2026. Os ids de modelo mudam com frequência; para usar outro, passe
`--model` ou preencha a variável do modelo (por exemplo `OPENAI_MODEL`) no
`.env`.

Dez dos treze arquivos têm o mesmo código e mudam só no bloco curto
"Provider settings" do topo, porque essas APIs copiam o formato Chat
Completions da OpenAI. Claude, Gemini e Perplexity têm formato próprio, e
os arquivos deles convertem as mensagens no padrão `{"role", "content"}`
para você.

## Requisitos

Python 3.9 ou mais recente. Os scripts da raiz usam só a biblioteca
padrão. A pasta opcional [`sdk/`](sdk/) usa os pacotes oficiais listados
em `sdk/requirements.txt`.

Você precisa de uma chave de API do provedor que for usar (os links estão
no [`.env.example`](.env.example)). O uso da API é cobrado pelo provedor,
à parte de qualquer assinatura do ChatGPT, do Claude ou do Gemini. O
Ollama roda no seu computador e não pede chave.

## Começo rápido

```bash
git clone https://github.com/LucasFerrazSEO/llm-api-python.git
cd llm-api-python
cp .env.example .env        # no Windows: copy .env.example .env
```

Abra o `.env`, cole a chave na linha certa (por exemplo
`OPENAI_API_KEY=sk-...`) e rode:

```bash
python chatgpt.py "Explique recursão em uma frase"
```

Os scripts leem a chave do ambiente ou de um arquivo `.env` na pasta atual
ou na pasta do script. Uma variável que já existe no ambiente sempre vale
mais que o arquivo.

## Linha de comando

Todos os scripts da raiz aceitam as mesmas opções:

```bash
python claude.py "Seu prompt"                          # envia e mostra a resposta
python claude.py --stream "Seu prompt"                 # mostra a resposta enquanto é gerada
python claude.py --chat                                # conversa com histórico (/reset, /exit)
python claude.py --system "Responda em inglês" "Oi"    # prompt de sistema
python claude.py --model claude-sonnet-5 "Oi"          # outro modelo
python claude.py --max-tokens 300 --temperature 0.2 "Oi"
python claude.py --json "Oi"                           # mostra o JSON bruto da resposta
cat notas.txt | python claude.py -                     # lê o prompt da entrada padrão
```

Opções extras: `gemini.py --search` embasa a resposta na Busca Google, e
`perplexity.py --preset fast|low|medium|high|xhigh` escolhe a profundidade
da pesquisa na web. Quando o provedor devolve fontes da web (Perplexity e
Gemini com `--search`), elas aparecem numeradas depois da resposta.

## Uso no seu código

Cada arquivo também é um módulo com três funções:

```python
from chatgpt import ask, send, complete

# um prompt, resposta em texto
print(ask("Escreva um slogan para uma padaria"))

# uma conversa, com prompt de sistema
historico = [
    {"role": "user", "content": "Meu nome é Ana."},
    {"role": "assistant", "content": "Prazer, Ana."},
    {"role": "user", "content": "Qual é o meu nome?"},
]
print(send(historico, system="Responda de forma breve"))

# streaming, com as fontes e a resposta bruta
texto, fontes, bruto = complete(historico, stream=True, on_token=lambda t: print(t, end=""))
```

As mesmas chamadas funcionam com `claude`, `gemini`, `grok` e os outros
arquivos, então trocar de provedor é trocar o import. Uma falha levanta
`APIError` com mensagem legível.

## Versões com o SDK oficial

Para quem prefere as bibliotecas oficiais, a pasta [`sdk/`](sdk/) faz o
mesmo envio e recebimento com elas:

```bash
pip install -r sdk/requirements.txt
python sdk/chatgpt_sdk.py "Oi"                     # SDK da OpenAI, Responses API
python sdk/claude_sdk.py --stream "Oi"             # SDK da Anthropic
python sdk/gemini_sdk.py --search "Oi"             # SDK Google Gen AI
python sdk/openai_compatible_sdk.py deepseek "Oi"  # SDK da OpenAI apontado para outro provedor
```

O `openai_compatible_sdk.py` mostra que o SDK da OpenAI funciona com Grok,
DeepSeek, Mistral, Groq, OpenRouter, Qwen, Kimi, Maritaca e Ollama
trocando só o `base_url`, a chave e o modelo.

## O mesmo prompt em várias IAs

O [`examples/ask_many.py`](examples/ask_many.py) envia o mesmo prompt a
todos os provedores com chave no `.env` e mostra cada resposta com o tempo
que levou:

```bash
python examples/ask_many.py "Explique recursão em uma frase"
python examples/ask_many.py --only claude,gemini,grok "Olá"
```

## Erros

Os scripts transformam os erros HTTP em mensagens curtas: chave inválida
ou ausente (401, e o 400 que Gemini e Grok devolvem para chave errada),
sem créditos (402, ou o `insufficient_quota` da OpenAI), sem permissão
(403), modelo ou URL errados (404) e limite de uso (429). Limite de uso,
erro de servidor e o 529 da Anthropic (sobrecarga) são tentados de novo
até três vezes, respeitando o `Retry-After`. Falta de crédito não é
repetida. A mensagem original do provedor sempre aparece abaixo do
resumo.

Cada arquivo também aceita uma variável que troca a URL (`OPENAI_API_URL`,
`ANTHROPIC_API_URL`, `GEMINI_API_URL` e assim por diante), útil para um
proxy, um gateway, outra região (Qwen) ou a nuvem do Ollama.

## Testes

```bash
python -m unittest discover -s tests -v
```

Os testes sobem um servidor falso local que responde nos formatos da
OpenAI, do Claude, do Gemini e da Perplexity, então não pedem chave nem
gastam crédito. Eles cobrem o formato da requisição de cada arquivo, o
streaming, as fontes, o prompt de sistema, as novas tentativas, as
mensagens de erro, a leitura do `.env` e o histórico do chat. O GitHub
Actions roda os testes no Python 3.9 e no 3.13.

## Perguntas frequentes

**A API é igual ao aplicativo do ChatGPT, do Claude ou do Gemini?**
Não. A API entrega o modelo; os aplicativos somam recursos próprios,
memória e configurações por cima. A busca na web, por exemplo, vem ligada
na Perplexity, é opcional no Gemini (`--search`) e fica desligada nos
outros scripts daqui, então a resposta sobre fatos recentes pode ser
diferente da que o aplicativo mostra.

**Preciso instalar alguma coisa?**
Não para os scripts da raiz: basta ter Python 3.9 ou mais recente. Só a
pasta `sdk/` pede `pip install`.

**Posso copiar só um arquivo para o meu projeto?**
Pode. Cada arquivo da raiz é independente: o `claude.py` não depende de
nenhum outro arquivo deste repositório. O único arquivo em comum é o
`.env`, e você pode definir as variáveis direto no ambiente.

**Qual provedor é gratuito?**
O Ollama roda modelos no seu computador sem custo. Alguns provedores têm
plano gratuito ou crédito de teste, e essas ofertas mudam; confira a
página de preços de cada um.

**Qual é a diferença entre Grok e Groq?**
Grok é o modelo da xAI (`grok.py`). Groq é uma empresa que serve modelos
abertos como o Llama com muita velocidade (`groq.py`).

## Limitações

Os ids de modelo padrão e os endpoints foram conferidos em 28/09/2026 e
vão envelhecer; a variável do modelo no `.env` permite atualizar sem
mexer no código. Os scripts enviam e recebem só texto: imagem, áudio,
envio de arquivo e chamada de ferramentas ficam fora do escopo. O
`perplexity.py` usa a Agent API porque a Perplexity encerrou o suporte ao
endpoint Sonar Chat Completions em 27/09/2026. Os arquivos `groq.py`,
`mistral.py` e `ollama.py` têm o mesmo nome dos pacotes pip dessas
empresas; se você instalar o pacote oficial na mesma pasta, renomeie o
arquivo antes de importar. Os testes offline comprovam o tratamento de
requisição e resposta contra um servidor falso que segue o formato
documentado de cada API; eles não substituem uma chamada real com a sua
chave.

## Contribuir

Relatos de erro e novos provedores são bem-vindos pelas [Issues do GitHub](https://github.com/LucasFerrazSEO/llm-api-python/issues).
Um novo provedor compatível com a OpenAI precisa só de uma cópia do
`chatgpt.py` com um novo bloco "Provider settings" e uma linha nos
testes.

## Autor

[Lucas Ferraz](https://lucasferraz.com) é especialista em SEO, criação de sites e Generative Engine Optimization e fundador da [Lucas Ferraz SEO](https://lucasferrazseo.com).

## Licença

MIT. Veja [LICENSE](LICENSE).
