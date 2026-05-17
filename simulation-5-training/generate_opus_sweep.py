"""
Sim 6: Generate PDA training data with Opus 4.6 as teacher.
Perspective sweep: CoT (1), 2-PDA, 3-PDA, 4-PDA, 5-PDA on same questions.
Uses Claude subagents via SSH.
"""

import json, re, time, random, subprocess, sys
from pathlib import Path
from datasets import load_dataset

SEED = 42
N_QUESTIONS = 200

# Worker perspectives pool (draw N for each sweep level)
PERSPECTIVES = [
    ("methodical", "You are a careful, methodical math solver. Break the problem into small steps. Double-check each calculation. Show your work clearly. End with #### <number>"),
    ("creative", "You are a creative problem solver who looks for shortcuts, patterns and elegant approaches. Find the most efficient solution path. End with #### <number>"),
    ("skeptical", "You are a skeptical math reviewer. Consider edge cases, off-by-one errors, and common mistakes. Verify assumptions before calculating. End with #### <number>"),
    ("algebraic", "You are an algebraic thinker. Set up equations and solve symbolically before plugging in numbers. Use variables to track quantities. End with #### <number>"),
    ("estimator", "You are a estimation-first solver. Start with a rough estimate of the answer, then work through the precise calculation. Compare your result to your estimate as a sanity check. End with #### <number>"),
]

COT_SYSTEM = "You are an expert math problem solver. Think step by step. Show your work clearly, check your calculations, and consider edge cases. End with #### <number>"

MERGE_TEMPLATE = """You are a math answer synthesizer. Below are {n} different solutions to the same problem.
Analyze all approaches:
- If they agree, confirm the answer with brief justification.
- If they disagree, identify which has correct reasoning.
- Do NOT just pick the majority -- evaluate the actual math.

End with #### <number>

Question: {question}

{solutions}

Synthesize the best answer:"""


def call_opus(prompt):
    """Call Claude Opus via the local API."""
    # Use the Claude CLI in non-interactive mode
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--model", "claude-opus-4-6"],
            capture_output=True, text=True, timeout=120,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception as e:
        print(f"  Opus error: {e}")
        return ""


def call_opus_with_system(system, user):
    """Call Opus with a system prompt baked into the user message."""
    prompt = f"{system}\n\nProblem: {user}"
    return call_opus(prompt)


def run_cot(question):
    """Single-pass CoT."""
    return call_opus_with_system(COT_SYSTEM, question)


def run_pda(question, n_workers):
    """Run N PDA workers + merge."""
    workers = PERSPECTIVES[:n_workers]
    outputs = []
    for name, system in workers:
        output = call_opus_with_system(system, question)
        outputs.append((name, output))
        time.sleep(0.5)

    # Build merge prompt
    solutions = ""
    for i, (name, output) in enumerate(outputs):
        solutions += f"\nSolution {i+1} ({name}):\n{output}\n"

    merge_prompt = MERGE_TEMPLATE.format(
        n=n_workers, question=question, solutions=solutions
    )
    merged = call_opus(merge_prompt)
    time.sleep(0.5)
    return merged, outputs


def extract_answer(text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return float(match.group(1).replace(",", ""))
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    for n in reversed(numbers):
        cleaned = n.replace(",", "").strip()
        if cleaned and cleaned != "-":
            try:
                return float(cleaned)
            except:
                continue
    return None


def extract_gsm8k_answer(answer_text):
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_text)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def main():
    random.seed(SEED)

    print("Loading GSM8K training set...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:N_QUESTIONS]

    # Output files
    outputs = {
        "cot": Path("opus_cot_gsm8k.jsonl"),
        "pda2": Path("opus_pda2_gsm8k.jsonl"),
        "pda3": Path("opus_pda3_gsm8k.jsonl"),
        "pda4": Path("opus_pda4_gsm8k.jsonl"),
        "pda5": Path("opus_pda5_gsm8k.jsonl"),
    }

    # Count existing for resume
    existing = {}
    for key, path in outputs.items():
        if path.exists():
            with open(path) as f:
                existing[key] = sum(1 for _ in f)
        else:
            existing[key] = 0

    # Open files for append
    files = {key: open(path, "a") for key, path in outputs.items()}

    try:
        for i, idx in enumerate(indices):
            item = ds[idx]
            question = item["question"]
            gt = extract_gsm8k_answer(item["answer"])
            if gt is None:
                continue

            print(f"\n[{i+1}/{N_QUESTIONS}] {question[:60]}...")

            # CoT
            if i >= existing.get("cot", 0):
                print("  CoT...", end=" ", flush=True)
                cot_out = run_cot(question)
                cot_ans = extract_answer(cot_out)
                cot_ok = cot_ans is not None and cot_ans == gt
                print(f"{'OK' if cot_ok else 'WRONG'} ({cot_ans})")
                files["cot"].write(json.dumps({
                    "question": question, "gt_answer": gt,
                    "reasoning": cot_out, "answer": cot_ans,
                    "correct": cot_ok, "method": "cot",
                }) + "\n")
                files["cot"].flush()

            # PDA sweep
            for n in [2, 3, 4, 5]:
                key = f"pda{n}"
                if i >= existing.get(key, 0):
                    print(f"  PDA-{n}...", end=" ", flush=True)
                    merged, workers = run_pda(question, n)
                    pda_ans = extract_answer(merged)
                    pda_ok = pda_ans is not None and pda_ans == gt
                    print(f"{'OK' if pda_ok else 'WRONG'} ({pda_ans})")
                    files[key].write(json.dumps({
                        "question": question, "gt_answer": gt,
                        "pda_reasoning": merged, "pda_answer": pda_ans,
                        "correct": pda_ok, "method": f"pda-{n}",
                        "worker_reasonings": [w[1] for w in workers],
                        "worker_names": [w[0] for w in workers],
                    }) + "\n")
                    files[key].flush()

    finally:
        for f in files.values():
            f.close()

    # Summary
    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)
    for key, path in outputs.items():
        correct = total = 0
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                total += 1
                if d["correct"]:
                    correct += 1
        print(f"  {key}: {correct}/{total} ({100*correct/total:.1f}%)")


if __name__ == "__main__":
    main()
