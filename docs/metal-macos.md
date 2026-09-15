# Running FreeToken locally on Apple Silicon (Metal)

Runbook for the `feat/apple-metal-backend` branch on a Mac. The native engine
is CUDA-only; on macOS `ft serve` proxies the FreeToken API to Apple's
`mlx_lm.server` (or llama.cpp's `llama-server`). Nothing here needs CUDA.

## One-time setup

```bash
cd ~/Projects/FreeToken
git checkout feat/apple-metal-backend
uv venv --python 3.12            # must be a native arm64 python
file .venv/bin/python            # expect: Mach-O 64-bit executable arm64
uv pip install -e .              # core package only; CUDA deps are Linux-marked
uv pip install mlx-lm pytest
```

Optional second engine: `brew install llama.cpp` puts `llama-server` on PATH
for `--backend llama` with GGUF files.

## Start

The `Makefile` wraps the common commands; `make help` lists them.

```bash
make serve                              # Qwen3-14B-4bit on port 1919
make serve-8b                           # Qwen3-8B-4bit, about 2x the tokens/sec
make serve MODEL=<repo> PORT=<port>     # anything else
make serve-bg                           # background, logs to /tmp/ft-metal.log
make stop                               # stop the server and its upstream engine
```

Or drive it directly:

```bash
source .venv/bin/activate
ft serve --model mlx-community/Qwen3-0.6B-4bit --port 1919
```

First run downloads the checkpoint into `~/.cache/huggingface/hub`. The
server is usable when `/health` reports `"maintenance": "serving"`:

```bash
curl -s localhost:1919/health
```

Run it in the background and keep the log:

```bash
nohup ft serve --model mlx-community/Qwen3-0.6B-4bit --port 1919 > /tmp/ft-metal.log 2>&1 &
```

`scripts/start-metal.sh [model]` does the same plus attaches `ft shell`;
`NO_CHAT=1 scripts/start-metal.sh` starts the API only.

## Use

```bash
# OpenAI chat
curl -s localhost:1919/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen3-0.6B-4bit","messages":[{"role":"user","content":"Say hi."}],"max_tokens":64}'

# Anthropic messages (translated to the upstream chat route by the proxy)
curl -s localhost:1919/v1/messages -H 'content-type: application/json' \
  -d '{"model":"any","max_tokens":64,"messages":[{"role":"user","content":"Say hi."}]}'

# terminal chat
ft shell

# Claude Code against the local model
ANTHROPIC_BASE_URL=http://127.0.0.1:1919 ANTHROPIC_API_KEY=local claude
```

The `model` field in requests is ignored on the Metal path; the served model
is whatever `--model` was. `/v1/models` lists its short name.

## Stop and inspect

```bash
curl -s localhost:1919/v1/stats            # request counters
lsof -iTCP:1919 -sTCP:LISTEN               # who holds the port
pkill -f 'ft serve --model'; pkill -f mlx_lm.server
```

The proxy listens on 1919 and the upstream engine on a 190xx loopback port
(`--metal-port` to pin it). Both must be gone before restarting.

## Choosing a model

Any MLX-converted checkpoint from https://huggingface.co/mlx-community works.
The ceiling is not the machine's RAM but the GPU wired-memory limit:

```bash
sysctl iogpu.wired_limit_mb     # 0 or unset means the default, about 2/3 of RAM
```

On a 16 GB M1 with that limit at 13000, roughly 12.5 GB may be resident, and
whatever the OS and other apps need comes out of the same 16 GB. Weights are
only part of the bill: the KV cache grows linearly with context and at
agent-sized prompts it rivals the weights. Measured per-token KV cost and
totals at a 32k-token context:

| Model | Weights | KV at 32k | Total | Verdict |
| --- | --- | --- | --- | --- |
| `Qwen3-0.6B-4bit` | 0.4 GB | 0.9 GB | 1.3 GB | smoke tests only |
| `Qwen3-8B-4bit` | 4.3 GB | 4.5 GB | 8.8 GB | comfortable |
| `Qwen3-14B-4bit` | 7.7 GB | 5.0 GB | 12.7 GB | fits short prompts, not long ones |
| `gpt-oss-20b-MXFP4-Q8` | 11.2 GB | 1.5 GB | 12.7 GB | the ceiling |
| `Mistral-Small-3.2-24B-4bit` | 12.3 GB | 5.0 GB | 17.3 GB | does not fit |
| `Qwen3-30B-A3B-4bit` | 16.0 GB | 3.0 GB | 19.0 GB | does not fit |
| `Qwen3-32B-4bit` | 17.2 GB | 8.0 GB | 25.2 GB | does not fit |

Three things decide that table.

- **KV cache scales with context.** Qwen3-14B costs 160 KB per token, so a
  30k-token agent prompt adds 4.7 GB on top of the weights. gpt-oss-20b is the
  largest model that fits precisely because its 64-dim heads make KV cheap
  (48 KB per token). mlx-lm has no KV-quantization flag today, so the only
  lever is a shorter context or a smaller model.
- **Mixture-of-experts buys speed, not memory.** Qwen3-30B-A3B activates 3B
  parameters per token but all 30B must be resident: unified memory has no
  host side to stream experts from, which is the premise FreeToken's CUDA
  expert-offload design inverts. Going past 20B here needs an offload backend
  such as the `freetoken-mlx` fork.
- **Quantization sets the per-parameter cost.** MLX 4-bit lands near 0.53 GB
  per billion parameters including group scales.

Qwen3 models think by default. Send `"thinking": {"type": "disabled"}` on
`/v1/messages` (or `chat_template_kwargs: {"enable_thinking": false}` on the
OpenAI route) to skip the thinking block.

## Disk and the model cache

Checkpoints land in `~/.cache/huggingface/hub` and are never pruned
automatically, so the cache outgrows the models actually in use. Inspect it
before downloading anything large:

```bash
df -h /             # free space
hf cache ls         # every cached repo with its size and last use
```

Prune with the CLI, which removes the blobs as well as the snapshot links.
Both commands take `--dry-run`:

```bash
hf cache prune                          # detached revisions, incomplete downloads
hf cache rm model/<org>/<name>          # one repo, add --yes to skip the prompt
```

Deleting a `models--*` directory by hand works too, but deleting only a
snapshot leaves its blobs behind and reclaims nothing. Other runtimes keep
their own stores (Ollama in `~/.ollama`, LM Studio in `~/.lmstudio`); those
are separate from this cache and often larger.

## Throughput

Decode is memory-bandwidth bound, not compute bound. Every token reads the
whole weight set, so the ceiling is roughly the chip's bandwidth divided by the
weight bytes, and MLX reaches about three quarters of it. A base M1 has
68 GB/s. Measured on one, with `make bench`:

| Model | Weights | Measured | Ceiling | Of ceiling |
| --- | --- | --- | --- | --- |
| `Qwen3-8B-4bit` | 4.3 GB | 12.2 tok/s | 15.9 tok/s | 77% |
| `Qwen3-14B-4bit` | 7.7 GB | 6.5 tok/s | 8.9 tok/s | 73% |

Two consequences. Halving the weights roughly doubles the rate, which is the
only large lever available. And a model that fits but nearly fills memory is
slow for a second reason: paging.

```bash
make serve-bg && make bench       # bench takes --weights-gb for the ceiling line
```

**Memory pressure costs about a quarter of the rate.** The same 14B model
measured 5.0 tok/s with 6 GB of swap in use and 6.5 tok/s with 1.6 GB. Check
`sysctl vm.swapusage` before benchmarking and close whatever is holding RAM;
browsers, virtual machines and Docker are the usual culprits.

**Speculative decoding does not help here.** Drafting Qwen3-14B with
Qwen3-0.6B (`mlx_lm server --draft-model`, 4 draft tokens) measured 4.1 tok/s
against a 6.5 tok/s control on the same engine, a 35% loss. On a bandwidth-
bound part the draft model's own weight reads and the verification pass cost
more than the accepted tokens save. This is why the Metal launcher does not
expose `--draft-model`.

## Tests

The Metal path is torch-free, so only these files run in this venv:

```bash
python -m pytest tests/server/test_metal_backend.py tests/server/test_serve_macos.py \
  tests/daemon/test_serve_command_platform.py tests/server/test_process_utils.py \
  tests/test_logger.py tests/test_shell_client.py tests/test_shell_tui.py -q
```

Everything else imports torch and is Linux/CUDA only.

## Troubleshooting

- **Stream shows only pings, no text, for minutes.** Check `sysctl vm.swapusage`.
  When the machine is deep in swap the engine is paged out and prefill crawls.
  Free memory (other node/cargo/claude processes) and retry. A resident 0.6B
  model prefills a 30k-token Claude Code prompt in under a minute.
- **First text takes a while even when resident.** The model is thinking.
  Disable thinking as above or use a larger model.
- **`no wheels with a matching Python implementation tag`.** The venv is x86_64
  under Rosetta. Recreate it from a native arm64 terminal.
- **`--backend llama` fails.** `llama-server` is not on PATH; `brew install llama.cpp`.
- **Port 1919 busy.** A previous server is still up; see Stop above.
- **`ModuleNotFoundError: torch`.** You ran a CUDA-only command or test file.
  `ft serve`, `ft serve-metal`, `ft shell` and the test files above never import torch.

## Branch upkeep

```bash
git fetch origin && git rebase origin/main     # upstream FlashML-org
git push cody feat/apple-metal-backend         # personal fork codyaverett/FreeToken
```
