"""
PDA Simulation 5: Generate training data for PDA distillation.
Takes GSM8K training questions, runs 3 PDA workers + merger,
and saves the merged reasoning as training examples.

Output format: JSONL with fields:
  - question: the math problem
  - pda_reasoning: merged reasoning from 3 workers
  - answer: extracted numerical answer
  - gt_answer: ground truth answer
  - correct: whether PDA got it right
  - worker_answers: list of 3 worker answers
  - worker_reasonings: list of 3 worker full outputs

Usage:
    python generate_training_data.py --n_examples 500 --output pda_training_data.jsonl
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

# For the distilled model: single-pass system prompt that mimics PDA reasoning
DISTILLED_SYSTEM = """You are a math problem solver who considers multiple approaches before answering. For each problem:
1. First, solve it step by step methodically.
2. Then, look for a more efficient approach or shortcut.
3. Finally, check for edge cases and common mistakes.
4. Synthesize the best answer from your analysis.

End with #### <number>"""


def call_model(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=1024,
            extra_body={"transforms": ["middle-out"]},
        )
        content = response.choices[0].message.content or ""
        # Strip thinking tags if present
        content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
        return content.strip()
    except Exception as e:
        print(f"  API error: {e}")
        return ""


def extract_answer(text: str) -> float | None:
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return float(match.group(1).replace(",", ""))
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
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
    if match:
        return float(match.group(1).replace(",", ""))
    raise ValueError(f"Could not extract answer from: {answer_text}")


def run_pda(question: str) -> dict:
    """Run 3 PDA workers + merger on a question."""
    # Workers (parallel in concept, sequential here)
    worker_outputs = []
    worker_answers = []
    for i, system in enumerate(PDA_WORKER_SYSTEMS):
        output = call_model(system, question)
        worker_outputs.append(output)
        answer = extract_answer(output)
        worker_answers.append(answer)
        time.sleep(0.3)  # rate limit

    # Merge
    merge_prompt = f"""Question: {question}

Solution 1 (methodical):
{worker_outputs[0]}

Solution 2 (creative):
{worker_outputs[1]}

Solution 3 (skeptical):
{worker_outputs[2]}

Synthesize the best answer:"""

    merged = call_model(PDA_MERGE_SYSTEM, merge_prompt)
    merged_answer = extract_answer(merged)
    time.sleep(0.3)

    return {
        "worker_reasonings": worker_outputs,
        "worker_answers": [str(a) for a in worker_answers],
        "pda_reasoning": merged,
        "pda_answer": merged_answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_examples", type=int, default=500)
    parser.add_argument("--output", type=str, default="pda_training_data.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_from", type=int, default=0,
                        help="Resume from this index (skip already generated)")
    args = parser.parse_args()

    print(f"Loading GSM8K training set...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    # Shuffle deterministically and take n_examples
    import random
    random.seed(args.seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:args.n_examples]

    # Load existing data if resuming
    existing = 0
    if args.start_from > 0 and Path(args.output).exists():
        with open(args.output) as f:
            existing = sum(1 for _ in f)
        print(f"Resuming from {existing} existing examples")

    correct = 0
    total = 0

    with open(args.output, "a") as f:
        for idx_num, ds_idx in enumerate(indices):
            if idx_num < args.start_from or idx_num < existing:
                continue

            item = ds[ds_idx]
            question = item["question"]
            gt_answer = extract_gsm8k_answer(item["answer"])

            print(f"\n[{idx_num + 1}/{args.n_examples}] Q: {question[:80]}...")

            result = run_pda(question)
            pda_correct = result["pda_answer"] is not None and result["pda_answer"] == gt_answer

            total += 1
            if pda_correct:
                correct += 1

            print(f"  GT: {gt_answer} | PDA: {result['pda_answer']} | "
                  f"{'OK' if pda_correct else 'WRONG'} | "
                  f"Running: {correct}/{total} ({100*correct/total:.1f}%)")

            entry = {
                "question": question,
                "gt_answer": gt_answer,
                "pda_reasoning": result["pda_reasoning"],
                "pda_answer": result["pda_answer"],
                "correct": pda_correct,
                "worker_answers": result["worker_answers"],
                "worker_reasonings": result["worker_reasonings"],
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

    print(f"\nDone! {correct}/{total} correct ({100*correct/total:.1f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
