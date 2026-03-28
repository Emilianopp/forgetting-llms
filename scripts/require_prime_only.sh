#!/usr/bin/env bash

if [[ "${ALLOW_LEGACY_VERL:-0}" != "1" ]]; then
    cat >&2 <<'EOF'
ERROR: This legacy VeRL script is disabled.

This repo now treats PRIME-RL as the only supported RL path.

Use one of:
  - bash scripts/run_all_tasks_prime_interactive_session.sh rl
  - python scripts/prime_rl_runner.py prime ...

If you intentionally need the old VeRL path for archaeology, set:
  ALLOW_LEGACY_VERL=1
EOF
    exit 1
fi
