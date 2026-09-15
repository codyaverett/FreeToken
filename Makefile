# Local Metal (Apple Silicon) workflow. See docs/metal-macos.md.
# Everything runs out of ./.venv; `make venv` creates it if missing.

MODEL ?= mlx-community/Qwen3-14B-4bit
PORT  ?= 1919
VENV  ?= .venv
PY    := $(VENV)/bin/python
FT    := $(VENV)/bin/ft
LOG   ?= /tmp/ft-metal.log

.PHONY: help venv serve serve-bg serve-8b serve-small stop restart health chat models cache test bench

help:
	@echo "make serve                 serve $(MODEL) on port $(PORT)"
	@echo "make serve MODEL=<repo>    serve another MLX checkpoint"
	@echo "make serve-8b              serve Qwen3-8B-4bit (faster, ~2x tok/s)"
	@echo "make serve-small           serve Qwen3-0.6B-4bit (smoke tests)"
	@echo "make serve-bg              serve in the background, log to $(LOG)"
	@echo "make stop / restart        stop the server and its upstream engine"
	@echo "make health / models       query a running server"
	@echo "make chat                  attach ft shell to a running server"
	@echo "make bench                 measure decode tokens/sec"
	@echo "make test                  run the torch-free Metal test files"
	@echo "make cache                 list the Hugging Face model cache"

venv: $(FT)
$(FT):
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(VENV) -e .
	uv pip install --python $(VENV) mlx-lm pytest

serve: venv
	$(FT) serve --model $(MODEL) --port $(PORT)

serve-8b:
	@$(MAKE) serve MODEL=mlx-community/Qwen3-8B-4bit

serve-small:
	@$(MAKE) serve MODEL=mlx-community/Qwen3-0.6B-4bit

serve-bg: venv
	@nohup $(FT) serve --model $(MODEL) --port $(PORT) > $(LOG) 2>&1 & \
	echo "serving $(MODEL) on :$(PORT), log $(LOG)"; \
	until curl -fsS -m 2 localhost:$(PORT)/health 2>/dev/null | grep -q serving; do sleep 2; done; \
	echo "ready"

stop:
	@pkill -f "ft serve --model" 2>/dev/null || true; \
	pkill -f mlx_lm.server 2>/dev/null || true; \
	pkill -f llama-server 2>/dev/null || true; \
	sleep 1; \
	lsof -iTCP:$(PORT) -sTCP:LISTEN >/dev/null 2>&1 && echo "port $(PORT) still held" || echo "stopped, port $(PORT) free"

restart: stop serve-bg

health:
	@curl -fsS -m 5 localhost:$(PORT)/health || echo "no server on :$(PORT)"

models:
	@curl -fsS -m 5 localhost:$(PORT)/v1/models || echo "no server on :$(PORT)"

chat:
	$(FT) shell --server http://127.0.0.1:$(PORT)

bench:
	@$(PY) scripts/bench_metal.py --port $(PORT)

test: venv
	$(PY) -m pytest tests/server/test_metal_backend.py tests/server/test_serve_macos.py \
	  tests/daemon/test_serve_command_platform.py tests/server/test_process_utils.py \
	  tests/test_logger.py tests/test_shell_client.py tests/test_shell_tui.py -q

cache:
	@$(VENV)/bin/hf cache ls
