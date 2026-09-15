# Local Metal (Apple Silicon) workflow. See docs/metal-macos.md.
# Everything runs out of ./.venv; `make venv` creates it if missing.

MODEL ?= mlx-community/Qwen3-14B-4bit
PORT  ?= 1919
VENV  ?= .venv
PY    := $(VENV)/bin/python
FT    := $(VENV)/bin/ft
LOG   ?= /tmp/ft-metal.log

.PHONY: help venv serve serve-bg serve-8b serve-8b-bg serve-small stop restart health chat models cache test bench logs agent-install agent-uninstall agent-status

help:
	@echo "make serve                 serve $(MODEL) on port $(PORT)"
	@echo "make serve MODEL=<repo>    serve another MLX checkpoint"
	@echo "make serve-8b              serve Qwen3-8B-4bit (faster, ~2x tok/s)"
	@echo "make serve-small           serve Qwen3-0.6B-4bit (smoke tests)"
	@echo "make serve-bg              serve in the background, log to $(LOG)"
	@echo "make serve-8b-bg           serve Qwen3-8B-4bit in the background"
	@echo "make logs                  follow the background server log"
	@echo "make stop / restart        stop the server and its upstream engine"
	@echo "make health / models       query a running server"
	@echo "make chat                  attach ft shell to a running server"
	@echo "make bench                 measure decode tokens/sec"
	@echo "make test                  run the torch-free Metal test files"
	@echo "make cache                 list the Hugging Face model cache"
	@echo "make agent-install         start the server at login and keep it up"
	@echo "make agent-uninstall       remove that login item"
	@echo "make agent-status          show whether it is loaded"

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

serve-8b-bg:
	@$(MAKE) serve-bg MODEL=mlx-community/Qwen3-8B-4bit

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

logs:
	@tail -f $(LOG)

cache:
	@$(VENV)/bin/hf cache ls

# Run the server as a per-user login item so it comes back after a reboot and
# restarts if it exits. Writes one plist to ~/Library/LaunchAgents and loads it
# into the user's own launchd domain; agent-uninstall reverses both steps.
# Override MODEL/PORT to pin what the login item serves.
AGENT_LABEL := org.freetoken.metal
AGENT_PLIST := $(HOME)/Library/LaunchAgents/$(AGENT_LABEL).plist

agent-install: venv
	@lsof -iTCP:$(PORT) -sTCP:LISTEN >/dev/null 2>&1 \
	  && { echo "port $(PORT) is already served; run 'make stop' first"; exit 1; } || true
	@mkdir -p $(HOME)/Library/LaunchAgents
	@printf '%s\n' \
	  '<?xml version="1.0" encoding="UTF-8"?>' \
	  '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' \
	  '<plist version="1.0">' \
	  '<dict>' \
	  '  <key>Label</key><string>$(AGENT_LABEL)</string>' \
	  '  <key>ProgramArguments</key>' \
	  '  <array>' \
	  '    <string>$(CURDIR)/$(FT)</string>' \
	  '    <string>serve</string>' \
	  '    <string>--model</string><string>$(MODEL)</string>' \
	  '    <string>--port</string><string>$(PORT)</string>' \
	  '  </array>' \
	  '  <key>WorkingDirectory</key><string>$(CURDIR)</string>' \
	  '  <key>RunAtLoad</key><true/>' \
	  '  <key>KeepAlive</key><true/>' \
	  '  <key>ProcessType</key><string>Background</string>' \
	  '  <key>StandardOutPath</key><string>$(LOG)</string>' \
	  '  <key>StandardErrorPath</key><string>$(LOG)</string>' \
	  '</dict>' \
	  '</plist>' > $(AGENT_PLIST)
	@plutil -lint $(AGENT_PLIST) >/dev/null
	@launchctl bootout gui/$$(id -u)/$(AGENT_LABEL) 2>/dev/null || true
	@launchctl bootstrap gui/$$(id -u) $(AGENT_PLIST)
	@echo "installed $(AGENT_PLIST)"
	@echo "serving $(MODEL) on :$(PORT), log $(LOG); remove with: make agent-uninstall"

agent-uninstall:
	@launchctl bootout gui/$$(id -u)/$(AGENT_LABEL) 2>/dev/null || true
	@rm -f $(AGENT_PLIST)
	@echo "removed $(AGENT_LABEL) and its plist"

agent-status:
	@launchctl print gui/$$(id -u)/$(AGENT_LABEL) 2>/dev/null \
	  | grep -E '^\s+(state|pid|last exit code) ' \
	  || echo "$(AGENT_LABEL) is not loaded"
