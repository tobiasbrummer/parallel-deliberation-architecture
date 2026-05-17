"""
PDA Simulation 4: Prompt-level PDA on GSM8K Benchmark
Compares baseline (single pass) vs PDA (3 workers + merge) using Qwen3-8B via OpenRouter.

Usage:
    python sim4_pda_benchmark.py --n_questions 10 --dry_run  # test with 10 questions
    python sim4_pda_benchmark.py --n_questions 200           # actual run
"""

import argparse
import json
import re
import time
import os
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset

# --- Config ---
MODEL = "qwen/qwen3-8b"
API_KEY = Path(os.path.expanduser("~/.config/api-keys/openrouter")).read_text().strip()
BASE_URL = "https://openrouter.ai/api/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- Prompts ---
BASELINE_SYSTEM = """You are a math problem solver. Solve the problem step by step, then give your final numerical answer on the last line in the format: #### <number>"""

PDA_WORKER_SYSTEMS = [
    """You are a careful, methodical math solver. Break the problem into small steps. Double-check each calculation. Show your work clearly. End with #### <number>""",
    """You are a creative problem solver who looks for shortcuts and patterns. Try to find the most efficient solution path. End with #### <number>""",
    """You are a skeptical math reviewer. Consider edge cases and common mistakes. Verify assumptions before calculating. End with #### <number>""",
]

PDA_MERGE_SYSTEM = """You are a math answer synthesizer. You will see 3 different solutions to the same math problem. Analyze all three approaches:
- If they agree on the answer, confirm it.
- If they disagree, identify which solution has the correct reasoning and pick that answer.
- Do NOT just pick the majority -- evaluate the actual math.

Give your final answer on the last line in the format: #### <number>"""


def call_model(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Call the model via OpenRouter."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
            extra_body={"transforms": ["middle-out"]},  # OpenRouter compression
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  API error: {e}")
        return ""


def extract_answer(text: str) -> float | None:
    """Extract the numerical answer after ####."""
    # Look for #### pattern
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return float(match.group(1).replace(",", ""))
    # Fallback: last number in text
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    for n in reversed(numbers):
        cleaned = n.replace(",", "").strip()
        if cleaned and cleaned != "-":
            try:
                return float(cleaned)
            except ValueError:
                continue
    return None


def extract_gsm8k_answer(answer_text: str) -> float:
    """Extract answer from GSM8K ground truth format."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
    if match:
        return float(match.group(1).replace(",", ""))
    return float('nan')


def run_baseline(question: str) -> dict:
    """Single-pass baseline."""
    response = call_model(BASELINE_SYSTEM, question, temperature=0.0)
    answer = extract_answer(response)
    return {"response": response, "answer": answer}


def run_pda(question: str, n_workers: int = 3) -> dict:
    """PDA: multiple workers + merge."""
    worker_responses = []
    for i in range(n_workers):
        system = PDA_WORKER_SYSTEMS[i % len(PDA_WORKER_SYSTEMS)]
        response = call_model(system, question, temperature=0.7)
        worker_responses.append(response)

    # Build merge prompt
    merge_input = f"Problem: {question}\n\n"
    for i, resp in enumerate(worker_responses):
        merge_input += f"--- Solution {i+1} ---\n{resp}\n\n"

    merge_response = call_model(PDA_MERGE_SYSTEM, merge_input, temperature=0.0)
    answer = extract_answer(merge_response)

    # Also extract individual worker answers for analysis
    worker_answers = [extract_answer(r) for r in worker_responses]

    return {
        "worker_responses": worker_responses,
        "worker_answers": worker_answers,
        "merge_response": merge_response,
        "answer": answer,
    }


def is_correct(predicted: float | None, ground_truth: float) -> bool:
    """Check if answer is correct (within small tolerance for floats)."""
    if predicted is None:
        return False
    return abs(predicted - ground_truth) < 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_questions", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true", help="Don't call API, just test loading")
    parser.add_argument("--output", type=str, default="sim4_results.json")
    parser.add_argument("--offset", type=int, default=0, help="Start from question N")
    parser.add_argument("--random", action="store_true", help="Random sample instead of sequential")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import random
    print(f"Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    all_questions = list(dataset)
    if args.random:
        random.seed(args.seed)
        questions = random.sample(all_questions, min(args.n_questions, len(all_questions)))
        print(f"Running on {len(questions)} randomly sampled questions (seed {args.seed})")
    else:
        questions = all_questions[args.offset:args.offset + args.n_questions]
        print(f"Running on {len(questions)} questions (offset {args.offset})")

    if args.dry_run:
        print(f"\nDry run -- first question:")
        print(f"  Q: {questions[0]['question'][:100]}...")
        print(f"  A: {extract_gsm8k_answer(questions[0]['answer'])}")
        return

    results = []
    baseline_correct = 0
    pda_correct = 0

    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = extract_gsm8k_answer(item["answer"])
        print(f"\n[{i+1}/{len(questions)}] Q: {question[:80]}...")
        print(f"  Ground truth: {ground_truth}")

        try:
            # Baseline
            baseline = run_baseline(question)
            b_correct = is_correct(baseline["answer"], ground_truth)
            baseline_correct += b_correct
            print(f"  Baseline: {baseline['answer']} {'OK' if b_correct else 'WRONG'}")

            # PDA
            pda = run_pda(question)
            p_correct = is_correct(pda["answer"], ground_truth)
            pda_correct += p_correct
            print(f"  PDA:      {pda['answer']} {'OK' if p_correct else 'WRONG'}")
            print(f"  Workers:  {pda['worker_answers']}")
        except Exception as e:
            print(f"  ERROR: {e} -- skipping")
            baseline = {"response": "", "answer": None}
            pda = {"worker_responses": [], "worker_answers": [], "merge_response": "", "answer": None}
            b_correct = False
            p_correct = False

        results.append({
            "index": args.offset + i,
            "question": question,
            "ground_truth": ground_truth,
            "baseline_answer": baseline["answer"],
            "baseline_correct": b_correct,
            "baseline_response": baseline["response"],
            "pda_answer": pda["answer"],
            "pda_correct": p_correct,
            "pda_worker_answers": pda["worker_answers"],
            "pda_merge_response": pda["merge_response"],
        })

        # Running stats
        n = i + 1
        print(f"  Running: Baseline {baseline_correct}/{n} ({baseline_correct/n:.1%}) | PDA {pda_correct}/{n} ({pda_correct/n:.1%})")

        # Small delay to be nice to the API
        time.sleep(0.5)

    # Final results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({len(questions)} questions)")
    print(f"  Baseline: {baseline_correct}/{len(questions)} ({baseline_correct/len(questions):.1%})")
    print(f"  PDA (n=3): {pda_correct}/{len(questions)} ({pda_correct/len(questions):.1%})")
    print(f"  Delta: {(pda_correct - baseline_correct)/len(questions):+.1%}")
    print(f"{'='*60}")

    # Save results
    output_path = Path(__file__).parent / args.output
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "model": MODEL,
                "n_questions": len(questions),
                "offset": args.offset,
                "n_workers": 3,
            },
            "summary": {
                "baseline_accuracy": baseline_correct / len(questions),
                "pda_accuracy": pda_correct / len(questions),
                "delta": (pda_correct - baseline_correct) / len(questions),
            },
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
