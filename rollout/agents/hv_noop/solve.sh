#!/bin/bash
# hv smoke agent: no LLM in the loop. Does literally what POST_TRAIN_BENCH_PROMPT=mock_test
# asks ("Just store {model} to final_model/ and exit") so that run_task.sh's plumbing --
# apptainer sandbox, check_cuda, timer, system_monitor, trace parsing, final_model
# collection, evaluate.py -> metrics.json -- can be exercised without an agent or any
# API credentials. Analogous to Harbor's `oracle` agent.
set -x
echo "hv_noop agent starting"
echo "PROMPT<<<"; echo "$PROMPT"; echo ">>>"
pwd; ls -la
bash timer.sh || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"

# The base model to copy is named in the prompt (mock_test.txt is exactly
# "Just store \`{model}\` to final_model/ and exit."), so read it from there.
python <<'PY'
import os, re, shutil, sys
from huggingface_hub import snapshot_download
prompt = os.environ.get("PROMPT", "")
# get_prompt.py escapes every backtick as \` , so a plain `([^`]+/[^`]+)` capture
# takes the backslash and yields 'Qwen/Qwen3-1.7B-Base\', which snapshot_download
# rejects with HFValidationError. Tolerate the escape, and refuse to guess: the
# control shipping a different base model than the recipe arms trained from would
# silently make the floor incomparable. Same fix as hv_recipe/solve.sh.
m = re.search(r"\\?`([^`\\]+/[^`\\]+)\\?`", prompt)
model_id = m.group(1).strip() if m else ""
if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", model_id):
    raise SystemExit(f"[hv_noop] FATAL: no valid base model id in $PROMPT (got {model_id!r})")
print("model_id from prompt:", model_id)
src = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
dst = os.path.join(os.getcwd(), "final_model")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst, symlinks=False)
print("copied", src, "->", dst)

#: `base` is the untrained control, and it must meet the eval on the same terms as the
#: trained arms. vLLM ignores `do_sample` and reads temperature/top_p/top_k from this
#: file, and Qwen3-*-Base names none of them -- so an unpinned control is sampled at the
#: library default of 1.0 while every hv_recipe cell is greedy, which makes "was this arm
#: trained" and "did this cell get the harness's decode" the same column. That is the
#: confound ptb_ops/make_greedy_shadow.py exists to remove; same four fields, same values.
#: The stop set is offered symmetrically for the same reason: training decides whether a
#: model EMITS the turn terminator, not which ones the server honours.
_gcp = os.path.join(dst, "generation_config.json")
try:
    import json
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(model_id)
    _tpl = os.path.join(os.getcwd(), "templates", "qwen3.jinja")
    if os.path.exists(_tpl):
        _tok.chat_template = open(_tpl).read()
    _SENT = "HVSTOPPROBE"
    _probe = _tok.apply_chat_template([{"role": "user", "content": "x"},
                                       {"role": "assistant", "content": _SENT}],
                                      tokenize=False)
    _stop = _probe.split(_SENT, 1)[1].strip() if _SENT in _probe else ""
    _ids = _tok.encode(_stop, add_special_tokens=False) if _stop else []
    with open(_gcp) as _f:
        _gc = json.load(_f)
    _prev = _gc.get("eos_token_id")
    _prev = _prev if isinstance(_prev, list) else ([] if _prev is None else [_prev])
    if len(_ids) == 1:
        _gc["eos_token_id"] = _ids + [i for i in _prev if i != _ids[0]]
    else:
        print(f"[hv_noop] WARNING: terminator {_stop!r} -> {_ids}, eos left as {_prev}")
    _gc.update({"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1})
    with open(_gcp, "w") as _f:
        json.dump(_gc, _f, indent=2)
    print(f"[hv_noop] decode pinned: {_gc}")
except Exception as _e:
    #: Fatal, not a warning. This used to print and continue, on the reasoning that an
    #: unpinned control is still a number -- but nothing downstream re-reads
    #: generation_config.json, so that number reaches the board indistinguishable from a
    #: pinned one, and it is sampled at 1.0 while every hv_recipe arm is greedy. "Was
    #: this arm trained" and "did this cell get the harness's decode" then become the
    #: same column, which is the exact confound the pin exists to remove. A missing
    #: control is a hole you can see; a mis-decoded one is a floor you cannot.
    raise SystemExit(f"[hv_noop] FATAL: could not pin decode ({_e!r}); refusing to ship "
                     f"a control whose decode does not match the hv_recipe arms")

print(sorted(os.listdir(dst)))
PY
NOOP_RC=$?
ls -la final_model || true
echo "hv_noop agent done rc=${NOOP_RC}"
# Propagate: without this the heredoc could raise, final_model/ be absent or
# unpinned, and run_task.sh still record `exit_code: 0 / status: exited normally`.
exit "${NOOP_RC}"
