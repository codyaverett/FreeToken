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
On 16 GB, 4-bit models up to about 8B parameters fit; larger ones swap.

| Model | Size on disk | Notes |
| --- | --- | --- |
| `mlx-community/Qwen3-0.6B-4bit` | ~0.4 GB | smoke tests; thinks at length, weak answers |
| `mlx-community/Qwen3-8B-4bit` | ~4.5 GB | good general model for 16 GB |

Qwen3 models think by default. Send `"thinking": {"type": "disabled"}` on
`/v1/messages` (or `chat_template_kwargs: {"enable_thinking": false}` on the
OpenAI route) to skip the thinking block.

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
