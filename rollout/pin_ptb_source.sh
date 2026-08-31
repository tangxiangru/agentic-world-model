#!/bin/bash
# Create one immutable, lightweight PTB source snapshot for a running pack.
set -euo pipefail

SOURCE="${1:?usage: pin_ptb_source.sh <prepared-ptb> <new-snapshot>}"
TARGET="${2:?usage: pin_ptb_source.sh <prepared-ptb> <new-snapshot>}"
SOURCE="$(cd "${SOURCE}" && pwd)"
[[ "${TARGET}" = /* ]] && [ "${TARGET}" != / ] && [ "${TARGET}" != "${TARGET%/*}" ] || {
    echo "FATAL: PTB snapshot target must be a specific absolute path" >&2
    exit 2
}
[ ! -e "${TARGET}" ] && [ ! -L "${TARGET}" ] || {
    echo "FATAL: refusing to reuse PTB source snapshot: ${TARGET}" >&2
    exit 2
}
git -C "${SOURCE}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "FATAL: prepared PTB source is not a Git checkout: ${SOURCE}" >&2
    exit 2
}
for path in .env src/run_task.sh \
    src/eval/general/prompt_fulltraj.txt \
    src/eval/general/prompt_wm.txt \
    src/eval/general/prompt_wm_fulltraj.txt \
    src/eval/general/prompt_wm_smoke.txt \
    src/eval/general/prompt_wm_fulltraj_smoke.txt \
    src/eval/tasks/gsm8k/test_data.json; do
    [ -f "${SOURCE}/${path}" ] && [ ! -L "${SOURCE}/${path}" ] || {
        echo "FATAL: prepared PTB source lacks regular study file: ${path}" >&2
        exit 2
    }
done

mkdir -p "$(dirname "${TARGET}")"
STAGE="$(mktemp -d "${TARGET}.tmp.XXXXXX")"
# `--shared` references the private checkout's object store but checks out only
# tracked source; ignored 25 GB container images remain at the .env site path.
git clone --quiet --shared "${SOURCE}" "${STAGE}"
install -m 0600 "${SOURCE}/.env" "${STAGE}/.env"
install -m 0755 "${SOURCE}/src/run_task.sh" "${STAGE}/src/run_task.sh"
for prompt in prompt_fulltraj.txt prompt_wm.txt prompt_wm_fulltraj.txt \
    prompt_wm_smoke.txt prompt_wm_fulltraj_smoke.txt; do
    install -m 0644 "${SOURCE}/src/eval/general/${prompt}" \
        "${STAGE}/src/eval/general/${prompt}"
done
# Generated benchmark test copies are Git-ignored, so the shared clone above
# cannot carry them.  Copy the exact setup-attested GSM8K fixture explicitly.
install -D -m 0600 "${SOURCE}/src/eval/tasks/gsm8k/test_data.json" \
    "${STAGE}/src/eval/tasks/gsm8k/test_data.json"
for agent in hv_recipe hv_noop claude_fulltraj_noawm claude_wm; do
    [ -d "${SOURCE}/agents/${agent}" ] && [ ! -L "${SOURCE}/agents/${agent}" ] || {
        echo "FATAL: prepared PTB source lacks study agent: ${agent}" >&2
        exit 2
    }
    # The Git checkout may contain an older tracked version of the same agent.
    # Stage-replace it so no stale auth/config file can survive into a cell.
    rm -rf "${STAGE}/agents/${agent}"
    mkdir -p "${STAGE}/agents/${agent}"
    cp -a "${SOURCE}/agents/${agent}/." "${STAGE}/agents/${agent}/"
done
bash -n "${STAGE}/src/run_task.sh"
mv "${STAGE}" "${TARGET}"
printf '%s\n' "${TARGET}"
