"""Generate single-pass Chain-of-Thought training data as control group for Sim 5c.
Uses the SAME questions as the PDA training data, but with one call instead of 3+merge.
This isolates whether PDA's multi-perspective approach matters, or just good reasoning."""

import json, re, time, random, os
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset

MODEL = "qwen/qwen3-8b"
API_KEY = Path(os.path.expanduser("~/.config/api-keys/openrouter")).read_text().strip()
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

COT_MATH_SYSTEM = """You are an expert math problem solver. Think step by step.
Show your work clearly, check your calculations, and consider edge cases.
Give your final answer after #### (for arithmetic) or in \\boxed{answer} format (for algebra)."""

COT_LOGIC_SYSTEM = """You are an expert logical reasoner. Think step by step.
Analyze the question carefully, consider each option, and explain your reasoning.
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

def extract_number(text):
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if m: return float(m.group(1).replace(",", ""))
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    for n in reversed(nums):
        c = n.replace(",", "").strip()
        if c and c != "-":
            try: return float(c)
            except: continue
    return None

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

def generate_cot(name, ds, indices, system, extract_fn, gt_fn, fmt_fn, outfile):
    existing = 0
    if Path(outfile).exists():
        with open(outfile) as f:
            existing = sum(1 for _ in f)
        print(f"Resuming {name} from {existing}")

    correct = total = 0
    with open(outfile, "a") as f:
        for i, idx in enumerate(indices):
            if i < existing: continue
            item = ds[idx]
            question = fmt_fn(item)
            gt = gt_fn(item)
            if gt is None: continue

            print(f"\n[CoT {name} {i+1}/{len(indices)}] {str(question)[:60]}...")
            reasoning = call_model(system, question)
            pred = extract_fn(reasoning)
            ok = normalize(pred) == normalize(gt)
            total += 1
            if ok: correct += 1
            print(f"  GT: {gt} | CoT: {pred} | {'OK' if ok else 'WRONG'} | {correct}/{total} ({100*correct/total:.1f}%)")
            time.sleep(0.3)

            f.write(json.dumps({
                "question": question, "gt_answer": str(gt),
                "cot_reasoning": reasoning, "cot_answer": str(pred) if pred else None,
                "correct": ok, "domain": name.lower(),
            }) + "\n")
            f.flush()

    print(f"\n{name} CoT done: {correct}/{total}")

def main():
    random.seed(42)

    # Use the SAME indices as PDA data for fair comparison
    # GSM8K
    print("=== GSM8K CoT ===")
    gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train")
    gsm8k_indices = list(range(len(gsm8k_ds)))
    random.shuffle(gsm8k_indices)
    gsm8k_indices = gsm8k_indices[:500]

    def gt_gsm(item):
        m = re.search(r'####\s*(-?[\d,]+\.?\d*)', item["answer"])
        return float(m.group(1).replace(",", "")) if m else None

    generate_cot("GSM8K", gsm8k_ds, gsm8k_indices, COT_MATH_SYSTEM,
                 extract_number, gt_gsm, lambda it: it["question"],
                 "cot_gsm8k_training.jsonl")

    # MATH
    print("\n=== MATH CoT ===")
    math_ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
    math_indices = list(range(len(math_ds)))
    random.shuffle(math_indices)
    math_indices = math_indices[:200]

    generate_cot("MATH", math_ds, math_indices, COT_MATH_SYSTEM,
                 extract_boxed, lambda it: extract_boxed(it["solution"]),
                 lambda it: it["problem"],
                 "cot_math_training.jsonl")

    # ARC
    print("\n=== ARC CoT ===")
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    arc_indices = list(range(len(arc_ds)))
    random.shuffle(arc_indices)
    arc_indices = arc_indices[:200]

    def fmt_arc(item):
        q = item["question"]
        for l, t in zip(item["choices"]["label"], item["choices"]["text"]):
            q += f"\n({l}) {t}"
        return q

    generate_cot("ARC", arc_ds, arc_indices, COT_LOGIC_SYSTEM,
                 extract_mc, lambda it: it["answerKey"], fmt_arc,
                 "cot_arc_training.jsonl")

if __name__ == "__main__":
    main()
