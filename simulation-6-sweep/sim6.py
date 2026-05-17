#!/usr/bin/env python3
"""
Sim 6: PDA Distillation Perspective Sweep — single training+eval run.

One run = (variant, seed). Train on Opus-distilled data for that variant,
eval on GSM8K. Saves adapter and per-question results to OUTPUT_DIR.

Usage:
    python sim6.py --variant cot --seed 42 --n-train 150 --n-eval 50
    python sim6.py --variant pda3 --seed 1337 --n-train 150 --n-eval 200

Dry-run (validate pipeline):
    python sim6.py --variant cot --seed 42 --n-train 150 --n-eval 50

Full sweep is just a shell loop over (variant, seed) — see sim6_sweep.sh.
"""

import argparse
import gc
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

# Single GPU, no distributed
for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
         "ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED", "ACCELERATE_MIXED_PRECISION"):
    os.environ.pop(k, None)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("ACCELERATE_BYPASS_DEVICE_MAP", "true")

import numpy as np
import torch

VARIANT_TO_FILE = {
    "cot":  ("opus_cot_gsm8k.jsonl",  "reasoning"),
    "pda2": ("opus_pda2_gsm8k.jsonl", "pda_reasoning"),
    "pda3": ("opus_pda3_gsm8k.jsonl", "pda_reasoning"),
    "pda4": ("opus_pda4_gsm8k.jsonl", "pda_reasoning"),
    "pda5": ("opus_pda5_gsm8k.jsonl", "pda_reasoning"),
}

MODEL_NAME = "unsloth/mistral-7b-v0.3-bnb-4bit"
MAX_SEQ_LEN = 2048


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sim6")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_correct(path: Path, reasoning_key: str, n: int, seed: int):
    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("correct", False):
                examples.append({
                    "question": d["question"],
                    "reasoning": d.get(reasoning_key, d.get("reasoning", "")),
                })
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:n]


def extract_number(text: str):
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return float(m.group(1).replace(",", ""))
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    for n in reversed(nums):
        c = n.replace(",", "").strip()
        if c and c != "-":
            try:
                return float(c)
            except ValueError:
                continue
    return None


def normalize(s):
    if s is None:
        return None
    s = str(s).strip().replace(" ", "").lower()
    try:
        return str(float(s))
    except ValueError:
        return s


def log_versions(logger):
    import transformers
    import peft
    import unsloth
    from importlib.metadata import version as _v
    try:
        import flash_attn
        fa_ver = flash_attn.__version__
    except Exception:
        fa_ver = "MISSING"
    logger.info("torch=%s transformers=%s peft=%s unsloth=%s flash_attn=%s",
                torch.__version__, transformers.__version__,
                _v("peft"), _v("unsloth"), fa_ver)
    logger.info("GPU=%s VRAM=%.1fGB bf16=%s",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9,
                torch.cuda.is_bf16_supported())


def train_adapter(variant: str, seed: int, n_train: int,
                  data_dir: Path, output_dir: Path, logger):
    save_path = output_dir / f"adapter-{variant}-seed{seed}"
    if (save_path / "adapter_model.safetensors").exists():
        logger.info("Adapter already exists at %s — skipping training", save_path)
        return save_path

    filename, reasoning_key = VARIANT_TO_FILE[variant]
    examples = load_correct(data_dir / filename, reasoning_key, n_train, seed)
    logger.info("Loaded %d training examples from %s", len(examples), filename)

    from unsloth import FastModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    logger.info("Loading base model %s", MODEL_NAME)
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model, r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    texts = [f"Question: {ex['question']}\n\nSolution: {ex['reasoning']}"
             for ex in examples]
    dataset = Dataset.from_list([{"text": t} for t in texts])

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=dataset, dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN, dataset_num_proc=2, packing=False,
        args=TrainingArguments(
            # batch=1 + grad_accum=8 keeps effective batch at 8 but halves
            # the VRAM peak. Required on RTX A4500 (20GB) with Mistral-7B 4bit;
            # safe on larger cards too.
            per_device_train_batch_size=1, gradient_accumulation_steps=8,
            warmup_steps=10, num_train_epochs=3, learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10, optim="adamw_8bit",
            weight_decay=0.01, lr_scheduler_type="linear",
            seed=seed, output_dir=str(save_path) + "-checkpoints",
            save_strategy="no", report_to="none",
            ddp_find_unused_parameters=False,
        ),
    )
    logger.info("Training start — variant=%s seed=%d examples=%d",
                variant, seed, len(examples))
    stats = trainer.train()
    logger.info("Training loss: %.4f", stats.training_loss)

    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    logger.info("Saved adapter -> %s", save_path)

    del model, tokenizer, trainer, dataset
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("VRAM after train cleanup: %.2fGB",
                torch.cuda.memory_allocated() / 1e9)
    return save_path


def evaluate_model(model, tokenizer, gsm8k_test, test_idx, tag, logger,
                   max_new_tokens=200):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopAfterAnswer(StoppingCriteria):
        """Stops generation ~8 tokens after the GSM8K '####' marker is seen,
        giving the model time to emit the full number without burning all
        max_new_tokens on follow-up text. Distilled students (esp. on Opus
        data) tend to keep generating new Q&A pairs otherwise."""
        def __init__(self, tokenizer, prompt_len: int):
            super().__init__()
            self.tokenizer = tokenizer
            self.prompt_len = prompt_len
            self.found_marker = False
            self.tokens_after = 0

        def __call__(self, input_ids, scores, **kwargs):
            if self.found_marker:
                self.tokens_after += 1
                return self.tokens_after >= 8
            start = max(self.prompt_len, input_ids.shape[1] - 30)
            text = self.tokenizer.decode(input_ids[0, start:],
                                          skip_special_tokens=True)
            if "####" in text:
                self.found_marker = True
            return False

    correct = total = 0
    per_question = []
    t_start = time.time()
    for i, idx in enumerate(test_idx):
        item = gsm8k_test[idx]
        m = re.search(r"####\s*(-?[\d,]+\.?\d*)", item["answer"])
        if not m:
            continue
        gt = float(m.group(1).replace(",", ""))

        prompt = f"Question: {item['question']}\n\nSolution: "
        ids = tokenizer(prompt, return_tensors="pt",
                        add_special_tokens=True).input_ids.to(model.device)
        stopping = StoppingCriteriaList([
            _StopAfterAnswer(tokenizer, prompt_len=ids.shape[1])
        ])
        with torch.inference_mode():
            out = model.generate(
                input_ids=ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )
        resp = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        pred = extract_number(resp)
        ok = pred is not None and normalize(str(pred)) == normalize(str(gt))
        per_question.append({
            "idx": int(idx), "gt": gt, "pred": pred,
            "correct": bool(ok), "response": resp[:500],
        })
        if ok:
            correct += 1
        total += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_start
            tps = total / elapsed
            logger.info("[%s] %d/%d: %d/%d (%.1f%%) | %.2f q/s",
                        tag, i + 1, len(test_idx), correct, total,
                        100 * correct / total, tps)

    elapsed = time.time() - t_start
    acc = round(100 * correct / total, 1) if total else 0.0
    logger.info("[%s] Final: %d/%d (%.1f%%) in %.0fs",
                tag, correct, total, acc, elapsed)
    return {
        "correct": correct, "total": total, "accuracy": acc,
        "elapsed_sec": elapsed, "per_question": per_question,
    }


def evaluate_run(variant: str, seed: int, adapter_path: Path,
                 n_eval: int, output_dir: Path, logger,
                 also_eval_base: bool):
    from unsloth import FastModel
    from datasets import load_dataset

    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(seed)
    test_idx = list(range(len(gsm8k_test)))
    rng.shuffle(test_idx)
    test_idx = test_idx[:n_eval]
    logger.info("Eval set: %d questions (seed=%d)", len(test_idx), seed)

    results = {}

    if also_eval_base:
        logger.info("Loading base model for baseline eval")
        model, tokenizer = FastModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
            full_finetuning=False,
        )
        FastModel.for_inference(model)

        sanity_ids = tokenizer("The capital of France is",
                                return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            sanity_out = model.generate(
                input_ids=sanity_ids, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        sanity_str = tokenizer.decode(sanity_out[0][sanity_ids.shape[-1]:],
                                       skip_special_tokens=True)
        logger.info("Sanity: %r", sanity_str)
        if "Paris" not in sanity_str:
            raise RuntimeError(f"Base model broken: {sanity_str!r}")

        logger.info("=== BASELINE ===")
        results["base"] = evaluate_model(model, tokenizer, gsm8k_test,
                                          test_idx, "Base", logger)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Loading adapter %s via Unsloth native loader", adapter_path)
    m, tok = FastModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    FastModel.for_inference(m)

    logger.info("=== %s DISTILLED (seed=%d) ===", variant.upper(), seed)
    results[variant] = evaluate_model(m, tok, gsm8k_test, test_idx,
                                       variant.upper(), logger)
    del m, tok
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=list(VARIANT_TO_FILE.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace"))
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip baseline eval (do once per seed, not per variant)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (e.g. eval-only on existing adapter)")
    args = parser.parse_args()

    run_tag = f"{args.variant}-seed{args.seed}"
    log_path = args.output_dir / "logs" / f"{run_tag}.log"
    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("Sim 6 run: variant=%s seed=%d n_train=%d n_eval=%d",
                args.variant, args.seed, args.n_train, args.n_eval)
    logger.info("=" * 60)

    log_versions(logger)
    seed_everything(args.seed)

    if args.skip_train:
        adapter_path = args.output_dir / f"adapter-{args.variant}-seed{args.seed}"
        if not (adapter_path / "adapter_model.safetensors").exists():
            logger.error("--skip-train but adapter not found at %s", adapter_path)
            sys.exit(2)
    else:
        adapter_path = train_adapter(
            args.variant, args.seed, args.n_train,
            args.data_dir, args.output_dir, logger,
        )

    results = evaluate_run(
        args.variant, args.seed, adapter_path,
        args.n_eval, args.output_dir, logger,
        also_eval_base=not args.skip_base,
    )

    out = {
        "config": {
            "variant": args.variant, "seed": args.seed,
            "n_train": args.n_train, "n_eval": args.n_eval,
            "model": MODEL_NAME,
        },
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_question"}
                    for k, v in results.items()},
        "per_question": {k: v.get("per_question", []) for k, v in results.items()},
    }
    out_path = args.output_dir / "results" / f"{run_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved results -> %s", out_path)

    base_acc = results.get("base", {}).get("accuracy")
    var_acc = results[args.variant]["accuracy"]
    if base_acc is not None:
        logger.info("Summary: base=%.1f%% %s=%.1f%% delta=%+.1fpp",
                    base_acc, args.variant, var_acc, var_acc - base_acc)
    else:
        logger.info("Summary: %s=%.1f%% (no baseline this run)",
                    args.variant, var_acc)


if __name__ == "__main__":
    main()
