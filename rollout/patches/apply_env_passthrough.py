#!/usr/bin/env python3
"""Pass an agent's explicitly listed non-secret runtime variables to PTB.

PostTrainBench launches agents with ``--cleanenv``. Study agents that use
Vertex therefore list only their routing variables in
``agents/<agent>/env_passthrough.txt``. Values are read from PTB's ordinary
launch environment; unlisted variables remain absent from the sandbox.
"""

from __future__ import annotations

import sys
from pathlib import Path


MARK = "# --- awm: explicit agent environment passthrough ---"
ANCHOR = 'echo "API keys provisioned for agent=${AGENT} task=${EVALUATION_TASK}: ${ALLOWED_API_KEYS[*]:-<none>}"\n'
EXEC_ENV = '        "${API_KEY_ENV_ARGS[@]}" \\\n'

BLOCK = f'''{MARK}
ENV_PASSTHROUGH_ARGS=()
ENV_PASSTHROUGH_FILE="agents/${{AGENT}}/env_passthrough.txt"
if [ -f "${{ENV_PASSTHROUGH_FILE}}" ]; then
    while IFS= read -r _env_name || [ -n "${{_env_name}}" ]; do
        case "${{_env_name}}" in
            ""|"#"*) continue ;;
        esac
        [[ "${{_env_name}}" =~ ^[A-Z][A-Z0-9_]*$ ]] || {{
            echo "ERROR: invalid environment name in ${{ENV_PASSTHROUGH_FILE}}: ${{_env_name}}" >&2
            exit 1
        }}
        if [ -n "${{!_env_name+x}}" ]; then
            ENV_PASSTHROUGH_ARGS+=(--env "${{_env_name}}=${{!_env_name}}")
        fi
    done < "${{ENV_PASSTHROUGH_FILE}}"
fi
'''


def apply(text: str) -> str:
    if MARK in text:
        return text
    if text.count(ANCHOR) != 1:
        raise SystemExit("run_task.sh: expected one API-key provisioning message")
    if text.count(EXEC_ENV) != 1:
        raise SystemExit("run_task.sh: expected one agent API-key env expansion")
    text = text.replace(ANCHOR, ANCHOR + "\n" + BLOCK, 1)
    return text.replace(EXEC_ENV, EXEC_ENV + '        "${ENV_PASSTHROUGH_ARGS[@]}" \\\n', 1)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "src/run_task.sh")
    old = path.read_text()
    new = apply(old)
    if new != old:
        path.write_text(new)
        print(f"{path}: patched (explicit agent environment passthrough)")
    else:
        print(f"{path}: already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
