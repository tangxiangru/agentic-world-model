#!/usr/bin/env python3
"""Completion-only SFT for gemma-3-4b-pt, rendered with the grader's own template.

Custom loop, because Gemma3's 262k vocab makes the standard Trainer path
materialise a ~26 GB fp32 logit tensor. Two consequences drive the design:
  * micro-batches are formed on a TOKEN budget, not a row count;
  * the lm_head is applied only at positions that carry a label, which is ~35%
    of tokens here, cutting both the logit memory and the head's FLOPs.
"""
from __future__ import annotations
import argparse, json, math, os, random, shutil, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForImageTextToText, get_cosine_schedule_with_warmup, set_seed

TEMPLATE_PATH = "templates/gemma3.jinja"
END_OF_TURN = "<end_of_turn>"


def build_features(rows, tok, tpl, max_len, eot_id, verbose=True):
    feats, dropped, lens = [], 0, []
    for r in rows:
        msgs = []
        if r.get("system"):
            msgs.append({"role": "system", "content": r["system"]})
        msgs.append({"role": "user", "content": r["prompt_user"]})
        prompt = tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False,
                                         add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        comp = r["completion"].strip()
        assert comp.endswith(END_OF_TURN), comp[-40:]
        c_ids = tok(comp, add_special_tokens=False)["input_ids"]
        assert c_ids[-1] == eot_id, c_ids[-3:]
        if len(p_ids) + len(c_ids) > max_len:
            dropped += 1
            continue
        feats.append({"input_ids": p_ids + c_ids,
                      "labels": [-100] * len(p_ids) + c_ids})
        lens.append(len(p_ids) + len(c_ids))
    if verbose:
        sl = sorted(lens)
        print(f"features={len(feats)} dropped={dropped} "
              f"({dropped / max(1, len(rows)):.3%}) p50={sl[len(sl)//2]} "
              f"p99={sl[int(len(sl)*0.99)]} max={sl[-1]} tokens={sum(sl)/1e6:.2f}M "
              f"label_tokens={sum(sum(1 for x in f['labels'] if x != -100) for f in feats)/1e6:.2f}M",
              flush=True)
    return feats


def make_batches(feats, token_budget, max_rows, seed):
    """Greedy length-sorted bucketing under a padded-token budget."""
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    batches, cur, curmax = [], [], 0
    for i in order:
        L = len(feats[i]["input_ids"])
        m = max(curmax, L)
        if cur and (m * (len(cur) + 1) > token_budget or len(cur) >= max_rows):
            batches.append(cur)
            cur, curmax = [i], L
        else:
            cur.append(i)
            curmax = m
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


def collate(feats, idxs, pad_id, device):
    n = max(len(feats[i]["input_ids"]) for i in idxs)
    ids, labels, mask = [], [], []
    for i in idxs:
        f = feats[i]
        k = n - len(f["input_ids"])
        ids.append(f["input_ids"] + [pad_id] * k)
        labels.append(f["labels"] + [-100] * k)
        mask.append([1] * len(f["input_ids"]) + [0] * k)
    t = lambda x: torch.tensor(x, device=device)
    return t(ids), t(labels), t(mask)


def loss_sum(model, ids, labels, mask):
    """Sum of token CE over labelled positions. lm_head only where it is needed."""
    out = model.model(input_ids=ids, attention_mask=mask)
    h = out[0][:, :-1, :]
    tgt = labels[:, 1:]
    sel = tgt != -100
    if sel.sum() == 0:
        return h.sum() * 0.0, 0
    logits = model.lm_head(h[sel]).float()
    return F.cross_entropy(logits, tgt[sel], reduction="sum"), int(sel.sum())


def save_bf16(model, tok, path, src_dir):
    os.makedirs(path, exist_ok=True)
    # transformers refuses to serialise a GenerationConfig with do_sample=False and
    # an explicit temperature, and that ValueError aborts save_pretrained BEFORE the
    # weights are written (it cost exp-04 a finished 1.4 h run). Hand it a config it
    # accepts; the greedy generation_config.json is copied in below and is what vLLM
    # actually reads.
    for k in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, k):
            setattr(model.generation_config, k, None)
    model.generation_config.do_sample = False
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    model.save_pretrained(path, state_dict=sd, safe_serialization=True)
    del sd
    tok.save_pretrained(path)
    for f in ("preprocessor_config.json", "processor_config.json", "generation_config.json"):
        s = os.path.join(src_dir, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(path, f))
    cfg_p = os.path.join(path, "config.json")
    cfg = json.load(open(cfg_p))
    cfg["torch_dtype"] = "bfloat16"
    cfg["dtype"] = "bfloat16"
    for sub in ("text_config", "vision_config"):
        if isinstance(cfg.get(sub), dict):
            cfg[sub]["torch_dtype"] = "bfloat16"
            cfg[sub]["dtype"] = "bfloat16"
    json.dump(cfg, open(cfg_p, "w"), indent=2)
    gc = json.load(open(os.path.join(path, "generation_config.json")))
    assert gc.get("temperature") == 0.0 and gc.get("eos_token_id") == [1, 106], gc
    assert os.path.exists(os.path.join(path, "model.safetensors.index.json"))
    print("saved", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--token-budget", type=int, default=8192)
    ap.add_argument("--max-rows-per-batch", type=int, default=24)
    ap.add_argument("--eff-batch", type=int, default=64, help="rows per optimizer step (approx)")
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--save-epoch-ends", action="store_true")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save a checkpoint every N optimizer steps")
    args = ap.parse_args()

    set_seed(args.seed)
    tpl = open(TEMPLATE_PATH).read()
    tok = AutoTokenizer.from_pretrained(args.model)
    eot_id = tok.convert_tokens_to_ids(END_OF_TURN)
    assert eot_id == 106, eot_id

    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rows = rows[:args.limit]
    feats = build_features(rows, tok, tpl, args.max_seq_len, eot_id)

    print("=" * 25, "RENDERED EXAMPLE", "=" * 25)
    print(tok.decode(feats[0]["input_ids"])[:900])
    print("...TARGET (loss is computed on exactly this)...")
    print(tok.decode([t for t in feats[0]["labels"] if t != -100])[-400:])
    print("=" * 68, flush=True)
    if args.dry_run:
        return

    dev = "cuda"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation=args.attn).to(dev)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for n, p in model.named_parameters():
        if "vision_tower" in n or "multi_modal_projector" in n:
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)

    n_epochs_int = max(1, math.ceil(args.epochs))
    all_steps, epoch_batches = [], []
    for e in range(n_epochs_int):
        b = make_batches(feats, args.token_budget, args.max_rows_per_batch, args.seed + e)
        epoch_batches.append(b)
    rows_per_epoch = len(feats)
    total_rows = int(rows_per_epoch * args.epochs)
    accum = max(1, round(args.eff_batch / (sum(len(b) for b in epoch_batches[0]) / len(epoch_batches[0]))))
    micro_total = sum(len(b) for b in epoch_batches)  # not used directly
    seq = []
    used = 0
    for e in range(n_epochs_int):
        for b in epoch_batches[e]:
            if used >= total_rows:
                break
            seq.append(b)
            used += len(b)
    n_steps = math.ceil(len(seq) / accum)
    if args.max_steps > 0:
        n_steps = min(n_steps, args.max_steps)
        seq = seq[:n_steps * accum]
    print(f"micro-batches={len(seq)} accum={accum} optimizer_steps={n_steps}", flush=True)

    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * n_steps), n_steps)
    save_at = set()
    if args.save_epoch_ends and args.epochs > 1:
        for k in range(1, int(math.ceil(args.epochs))):
            save_at.add(max(1, round(n_steps * k / args.epochs)))
    print("intermediate saves at steps", sorted(save_at), flush=True)

    model.train()
    t0 = time.time()
    step = 0
    run_loss, run_tok = 0.0, 0
    log = []
    for gi in range(n_steps):
        group = seq[gi * accum:(gi + 1) * accum]
        if not group:
            break
        denom = 0
        losses = []
        # count labelled tokens in the group first, so the loss is a true mean
        for idxs in group:
            denom += sum(sum(1 for x in feats[i]["labels"] if x != -100) - 0 for i in idxs)
        denom = max(1, denom)
        for idxs in group:
            ids, labels, mask = collate(feats, idxs, tok.pad_token_id, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls, ntok = loss_sum(model, ids, labels, mask)
            (ls / denom).backward()
            run_loss += float(ls.detach())
            run_tok += ntok
            del ids, labels, mask, ls
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0 or step == 1:
            el = time.time() - t0
            msg = (f"step {step}/{n_steps} loss {run_loss/max(1,run_tok):.4f} "
                   f"lr {sched.get_last_lr()[0]:.2e} {el/60:.1f}min "
                   f"eta {(el/step)*(n_steps-step)/60:.1f}min "
                   f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}G")
            print(msg, flush=True)
            log.append({"step": step, "loss": run_loss / max(1, run_tok)})
            run_loss, run_tok = 0.0, 0
        if step in save_at or (args.save_every and step % args.save_every == 0 and step < n_steps):
            model.config.use_cache = True
            save_bf16(model, tok, os.path.join(args.out, f"step{step}"), args.model)
            model.config.use_cache = False

    final = os.path.join(args.out, "final")
    model.config.use_cache = True
    save_bf16(model, tok, final, args.model)
    json.dump(log, open(os.path.join(args.out, "losslog.json"), "w"))
    print("saved", final, f"wall {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
