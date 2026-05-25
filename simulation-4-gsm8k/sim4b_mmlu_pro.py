"""
PDA Simulation 4b: Prompt-level PDA on MMLU-Pro Benchmark
Multiple choice with 10 options (A-J). Baseline ~66% for Qwen3-8B.

Backend is pluggable via PDA_BACKEND env var (openrouter | local_hf).
See model_backend.py for details.
"""

import argparse
import json
import re
import time
import random
import os
from pathlib import Path
from datasets import load_dataset

# Pluggable backend (OpenRouter API / local HF) -- see model_backend.py
from model_backend import call_model, describe_backend, BACKEND

BASELINE_SYSTEM = """You are an expert answering multiple-choice questions. Think step by step, then give your final answer as a single letter (A-J) on the last line in the format: Answer: X"""

PDA_WORKER_SYSTEMS = [
    """You are a careful, methodical expert. Analyze each option systematically. Eliminate clearly wrong answers first, then evaluate remaining options. Think step by step. End with: Answer: X""",
    """You are an expert who looks for patterns and shortcuts. Consider which answer "feels" most consistent with domain knowledge. Check for common traps and trick options. End with: Answer: X""",
    """You are a skeptical expert. Question assumptions in the question. Consider edge cases. Look for subtle distinctions between similar-looking options. End with: Answer: X""",
]

PDA_MERGE_SYSTEM = """You are an answer synthesizer for a multiple-choice question. You will see 3 expert analyses.
- If all agree, confirm the answer.
- If they disagree, evaluate which reasoning is strongest. Don't just pick the majority.
- Pay attention to which expert caught errors the others missed.

Give your final answer as: Answer: X"""


def format_question(item) -> str:
    """Format MMLU-Pro question with options."""
    q = item["question"]
    options = item["options"]
    formatted = f"{q}\n\n"
    for i, opt in enumerate(options):
        letter = chr(65 + i)  # A, B, C...
        formatted += f"{letter}. {opt}\n"
    return formatted


def extract_answer(text: str) -> str | None:
    """Extract answer letter from response."""
    # Look for "Answer: X" pattern
    match = re.search(r'Answer:\s*([A-Ja-j])', text)
    if match:
        return match.group(1).upper()
    # Fallback: last single letter A-J on its own
    matches = re.findall(r'\b([A-Ja-j])\b', text)
    if matches:
        return matches[-1].upper()
    return None


def answer_index_to_letter(idx: int) -> str:
    return chr(65 + idx)


def run_baseline(question_text: str) -> dict:
    response = call_model(BASELINE_SYSTEM, question_text, temperature=0.0)
    answer = extract_answer(response)
    return {"response": response, "answer": answer}


def run_pda(question_text: str) -> dict:
    workers = []
    for sys in PDA_WORKER_SYSTEMS:
        resp = call_model(sys, question_text, temperature=0.7)
        workers.append(resp)

    merge_input = f"Question:\n{question_text}\n\n"
    for i, w in enumerate(workers):
        merge_input += f"--- Expert {i+1} ---\n{w}\n\n"

    merge_resp = call_model(PDA_MERGE_SYSTEM, merge_input, temperature=0.0)
    answer = extract_answer(merge_resp)
    worker_answers = [extract_answer(w) for w in workers]

    return {
        "worker_responses": workers,
        "worker_answers": worker_answers,
        "merge_response": merge_resp,
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="sim4b_mmlu_pro_results.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("Loading MMLU-Pro...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    all_q = list(dataset)
    random.seed(args.seed)
    questions = random.sample(all_q, min(args.n_questions, len(all_q)))
    print(f"Running on {len(questions)} questions (seed {args.seed})")

    if args.dry_run:
        q = questions[0]
        print(f"\nCategory: {q['category']}")
        print(f"Q: {q['question'][:100]}...")
        print(f"Options: {len(q['options'])}")
        print(f"Answer: {answer_index_to_letter(q['answer_index'])}")
        return

    results = []
    b_correct = p_correct = 0

    for i, item in enumerate(questions):
        q_text = format_question(item)
        gt = answer_index_to_letter(item["answer_index"])
        cat = item["category"]
        print(f"\n[{i+1}/{len(questions)}] ({cat}) Q: {item['question'][:60]}...")
        print(f"  GT: {gt}")

        try:
            baseline = run_baseline(q_text)
            b_ok = baseline["answer"] == gt
            b_correct += b_ok
            print(f"  Baseline: {baseline['answer']} {'OK' if b_ok else 'WRONG'}")

            pda = run_pda(q_text)
            p_ok = pda["answer"] == gt
            p_correct += p_ok
            print(f"  PDA:      {pda['answer']} {'OK' if p_ok else 'WRONG'}")
            print(f"  Workers:  {pda['worker_answers']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            baseline = {"response": "", "answer": None}
            pda = {"worker_responses": [], "worker_answers": [], "merge_response": "", "answer": None}
            b_ok = p_ok = False

        results.append({
            "index": i, "category": cat,
            "question": item["question"], "ground_truth": gt,
            "baseline_answer": baseline["answer"], "baseline_correct": b_ok,
            "pda_answer": pda["answer"], "pda_correct": p_ok,
            "pda_worker_answers": pda.get("worker_answers", []),
        })

        n = i + 1
        print(f"  Running: B {b_correct}/{n} ({b_correct/n:.1%}) | PDA {p_correct}/{n} ({p_correct/n:.1%})")
        time.sleep(0.3)

    print(f"\n{'='*60}")
    print(f"FINAL ({len(questions)} questions)")
    print(f"  Baseline: {b_correct}/{len(questions)} ({b_correct/len(questions):.1%})")
    print(f"  PDA:      {p_correct}/{len(questions)} ({p_correct/len(questions):.1%})")
    print(f"  Delta:    {(p_correct-b_correct)/len(questions):+.1%}")

    # Category breakdown
    from collections import defaultdict
    cats = defaultdict(lambda: {"b": 0, "p": 0, "n": 0})
    for r in results:
        c = r["category"]
        cats[c]["n"] += 1
        cats[c]["b"] += r["baseline_correct"]
        cats[c]["p"] += r["pda_correct"]
    print(f"\nPer category:")
    for c, v in sorted(cats.items()):
        if v["n"] >= 5:
            print(f"  {c}: B {v['b']}/{v['n']} ({v['b']/v['n']:.0%}) | PDA {v['p']}/{v['n']} ({v['p']/v['n']:.0%})")

    # Disagreements
    disagree = pda_fixed = pda_broke = 0
    for r in results:
        if r["baseline_correct"] != r["pda_correct"]:
            disagree += 1
            if r["pda_correct"]: pda_fixed += 1
            else: pda_broke += 1
    print(f"\nDisagreements: {disagree} (fixed {pda_fixed}, broke {pda_broke})")
    print(f"{'='*60}")

    with open(Path(__file__).parent / args.output, "w") as f:
        json.dump({
            "config": {"backend": describe_backend(), "n": len(questions), "seed": args.seed, "benchmark": "MMLU-Pro"},
            "summary": {
                "baseline": b_correct/len(questions),
                "pda": p_correct/len(questions),
                "delta": (p_correct-b_correct)/len(questions),
            },
            "results": results,
        }, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
