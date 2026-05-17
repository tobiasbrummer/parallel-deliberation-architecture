"""Generate PDA training data for MATH and ARC-Challenge benchmarks."""

import argparse, json, re, time, random, os
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset

MODEL = "qwen/qwen3-8b"
API_KEY = Path(os.path.expanduser("~/.config/api-keys/openrouter")).read_text().strip()
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

MATH_WORKERS = [
    "You are a careful, methodical math solver. Break the problem into small steps. Double-check each calculation. Show your work clearly.",
    "You are a creative problem solver who looks for shortcuts, patterns and elegant approaches. Find the most efficient solution path.",
    "You are a skeptical math reviewer. Consider edge cases, off-by-one errors, and common mistakes. Verify assumptions before calculating.",
]

LOGIC_WORKERS = [
    "You are a systematic logical reasoner. Identify premises, apply deductive reasoning step by step, and state your conclusion clearly.",
    "You are an intuitive problem solver. Look for analogies, patterns, and shortcuts. Consider what the question is really testing.",
    "You are a devil's advocate. Consider why each answer choice might be wrong. Eliminate options systematically before choosing.",
]

MATH_MERGE = """You are a math answer synthesizer. Analyze all three solutions:
- If they agree, confirm the answer.
- If they disagree, identify which has correct reasoning.
- Do NOT just pick the majority -- evaluate the actual math.
Give your final answer in \\boxed{answer} format."""

LOGIC_MERGE = """You are a reasoning synthesizer. Analyze all three approaches:
- If they agree, confirm with brief justification.
- If they disagree, evaluate which reasoning is most sound.
State your final answer as: The answer is (X)"""

def call_model(system, prompt, temp=0.7):
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=temp, max_tokens=1024,
            extra_body={"transforms": ["middle-out"]},
        )
        c = r.choices[0].message.content or ""
        return re.sub(r'<think>.*?</think>\s*', '', c, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"  API error: {e}")
        time.sleep(3)
        return ""

def run_pda(question, workers, merge_system):
    outs = []
    for w in workers:
        outs.append(call_model(w, question))
        time.sleep(0.3)
    merge_prompt = f"Question: {question}\n\nSolution 1:\n{outs[0]}\n\nSolution 2:\n{outs[1]}\n\nSolution 3:\n{outs[2]}\n\nSynthesize:"
    merged = call_model(merge_system, merge_prompt)
    time.sleep(0.3)
    return merged, outs

def extract_boxed(text):
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    return m.group(1).strip() if m else None

def extract_mc(text):
    m = re.search(r'(?:answer is|Answer:?)\s*\(?([A-E])\)?', text, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'\(?([A-E])\)\s*$', text.strip())
    return m.group(1).upper() if m else None

def normalize(s):
    if s is None: return None
    s = str(s).strip().replace(" ", "").lower()
    if s.endswith("."): s = s[:-1]
    try: return str(float(s))
    except: return s

def generate(name, ds, indices, domain, workers, merge_sys, extract_fn, gt_fn, fmt_fn, outfile):
    existing = 0
    if Path(outfile).exists():
        with open(outfile) as f:
            existing = sum(1 for _ in f)
        print(f"Resuming {name} from {existing}")

    correct = 0
    total = 0
    with open(outfile, "a") as f:
        for i, idx in enumerate(indices):
            if i < existing: continue
            item = ds[idx]
            question = fmt_fn(item)
            gt = gt_fn(item)
            if gt is None: continue

            print(f"\n[{name} {i+1}/{len(indices)}] {str(question)[:60]}...")
            merged, workers_out = run_pda(question, workers, merge_sys)
            pred = extract_fn(merged)
            ok = normalize(pred) == normalize(gt)
            total += 1
            if ok: correct += 1
            print(f"  GT: {gt} | PDA: {pred} | {'OK' if ok else 'WRONG'} | {correct}/{total} ({100*correct/total:.1f}%)")

            f.write(json.dumps({
                "question": question, "gt_answer": str(gt),
                "pda_reasoning": merged, "pda_answer": str(pred) if pred else None,
                "correct": ok, "domain": name.lower(),
                "worker_reasonings": workers_out,
            }) + "\n")
            f.flush()

    print(f"\n{name} done: {correct}/{total}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_math", type=int, default=200)
    p.add_argument("--n_arc", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    # MATH
    print("Loading MATH...")
    math_ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
    math_idx = random.sample(range(len(math_ds)), args.n_math)
    generate("MATH", math_ds, math_idx, "math", MATH_WORKERS, MATH_MERGE,
             extract_boxed, lambda it: extract_boxed(it["solution"]),
             lambda it: it["problem"], "pda_math_training.jsonl")

    # ARC
    print("\nLoading ARC-Challenge...")
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_idx = random.sample(range(len(arc_ds)), args.n_arc)
    def fmt_arc(item):
        q = item["question"]
        for l, t in zip(item["choices"]["label"], item["choices"]["text"]):
            q += f"\n({l}) {t}"
        return q
    generate("ARC", arc_ds, arc_idx, "logic", LOGIC_WORKERS, LOGIC_MERGE,
             extract_mc, lambda it: it["answerKey"],
             fmt_arc, "pda_arc_training.jsonl")

if __name__ == "__main__":
    main()
