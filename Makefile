ENV_FILE ?= .env

# C = Corti green accent (rgb 198,246,82), B = bold, D = dim, R = reset; all
# empty when NO_COLOR is set or output is not a terminal ($(shell) captures
# stdout, so probe the inherited stderr).
ifeq ($(NO_COLOR)$(shell test -t 2 || echo notty),)
C := \033[38;2;198;246;82m
B := \033[1m
D := \033[2m
R := \033[0m
endif

.DEFAULT_GOAL := help
.PHONY: help install setup _setup-external

help: ## Show available targets
	@printf '\n  $(B)agent-evals$(R) $(D)· evaluation harness$(R)\n\n'
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## /{printf "  $(C)%-18s$(R) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf '\n'

# Opik ships in the base dependencies, so a plain sync is all a run needs —
# there is no extra to opt into. `setup` lists this as a prerequisite, so
# `make setup` installs before it configures; run it alone to just sync.
install: ## Install dependencies into .venv (uv sync)
	@printf '\n  $(D)syncing dependencies with uv…$(R)\n\n'
	@uv sync

# A numbered stepper. `setup` only *writes* the env file — no value is probed
# here; bad credentials surface at the run-start preflight.
setup: install ## Configure .env for a run (paste your Corti Console block)
	@printf '\n  $(B)agent-evals setup$(R)\n\n'
	@printf '  $(C)✓$(R) $(D)1 ·$(R) install    dependencies ready $(D)(.venv)$(R)\n'
	@$(MAKE) --no-print-directory _setup-external

# Step 2 (paste) and step 3 (opik) of the stepper. The Console block is
# everything a run needs — it names the tenant, is the fallback OAuth
# client, and supplies the judge bearer's ingredients. CORTI_ENVIRONMENT (eu/us)
# selects the judge endpoint; JUDGE_BASE_URL is written from it so US clients
# hit ai.us.corti.app and EU clients hit ai.eu.corti.app.
_setup-external:
	@printf '  $(C)◇$(R) install   $(C)◇$(R) paste   $(C)◇$(R) opik\n\n'; \
	printf '  $(C)▸$(R) $(D)2 ·$(R) paste      your Console block\n'; \
	printf '      $(D)Corti Console → your project → Developer quickstart → “Copy as .env”$(R)\n'; \
	printf '      $(D)paste the block; finish with an empty line (or ctrl-d)$(R)\n\n'; \
	tmp=$$(mktemp); \
	while IFS= read -r line; do \
		[ -z "$$line" ] && break; \
		case "$$line" in \
		CORTI_TENANT_NAME=*|CORTI_CLIENT_ID=*|CORTI_CLIENT_SECRET=*|CORTI_ENVIRONMENT=*) \
			printf '%s\n' "$$line" >> "$$tmp";; \
		*) printf '      $(D)ignored: %s$(R)\n' "$$line";; \
		esac; \
	done; \
	if ! grep -q '^CORTI_CLIENT_ID=.' "$$tmp" 2>/dev/null \
		|| ! grep -q '^CORTI_CLIENT_SECRET=.' "$$tmp" 2>/dev/null; then \
		printf '      $(B)✗$(R) the block must set CORTI_CLIENT_ID and CORTI_CLIENT_SECRET.\n'; \
		rm -f "$$tmp"; exit 1; \
	fi; \
	touch "$(ENV_FILE)"; \
	sed -i.bak '/^CORTI_/d' "$(ENV_FILE)"; rm -f "$(ENV_FILE).bak"; \
	cat "$$tmp" >> "$(ENV_FILE)"; rm -f "$$tmp"; \
	secret=$$(grep '^CORTI_CLIENT_SECRET=' "$(ENV_FILE)" | cut -d= -f2-); \
	printf '      $(C)✓$(R) wrote the Console block to %s $(D)· secret …%s$(R)\n' \
		"$(ENV_FILE)" "$$(printf %s "$$secret" | tail -c 4)"; \
	printf '  $(C)▸$(R) $(D)3 ·$(R) opik       send results to your own Opik? [y/N] $(C)›$(R) '; \
	read opik; \
	case "$$opik" in \
	y|Y|yes) \
		printf '      OPIK_URL_OVERRIDE (your Opik base URL): '; read url; \
		printf '      OPIK_API_KEY: '; read key; \
		printf '      OPIK_PROJECT_NAME [Agents]: '; read proj; \
		for pair in "OPIK_URL_OVERRIDE=$$url" "OPIK_API_KEY=$$key" "OPIK_PROJECT_NAME=$$proj"; do \
			name=$${pair%%=*}; val=$${pair#*=}; \
			[ -z "$$val" ] && continue; \
			sed -i.bak "/^$$name=/d" "$(ENV_FILE)"; rm -f "$(ENV_FILE).bak"; \
			printf '%s\n' "$$pair" >> "$(ENV_FILE)"; \
		done; \
		printf '      $(C)✓$(R) Opik output redirected to %s\n' "$$url"; \
		;; \
	*) printf '      $(D)results stay in local result files (no remote Opik)$(R)\n'; \
	esac; \
	if grep -qE '^JUDGE_API_KEY=.' "$(ENV_FILE)"; then \
		printf '      $(D)note: JUDGE_API_KEY in %s overrides the pasted block for judge verdicts$(R)\n' "$(ENV_FILE)"; \
	fi; \
	env_val=$$(grep '^CORTI_ENVIRONMENT=' "$(ENV_FILE)" | cut -d= -f2-); \
	env_val=$${env_val:-eu}; \
	judge_url="https://ai.$$env_val.corti.app/v1"; \
	sed -i.bak '/^JUDGE_BASE_URL=/d' "$(ENV_FILE)"; rm -f "$(ENV_FILE).bak"; \
	printf 'JUDGE_BASE_URL=%s\n' "$$judge_url" >> "$(ENV_FILE)"; \
	printf '      $(C)✓$(R) judge endpoint %s $(D)· from CORTI_ENVIRONMENT=%s$(R)\n' "$$judge_url" "$$env_val"; \
	printf '\n  $(C)$(B)✓ Setup complete$(R)\n'; \
	printf '      uv run agent-evals run $(C)<suite.yaml>$(R) --env %s\n\n' "$$env_val"
