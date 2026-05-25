#!/usr/bin/env python3
"""Recover per-item Sim 4b MMLU-Pro results from the run log.

The original `sim4b_mmlu_pro_200.json` was written with all
`baseline_answer`/`pda_answer` fields as `null` because the run was
aborted at item 120/200 (API-cost decision) and the final write path
serialized stub values instead of the live data.

The actual answers and per-item correctness are captured in
`sim4b_run.log` line-by-line. This script parses the log and writes
`sim4b_mmlu_pro_partial_recovered.json` with the recovered fields, plus
a top-level summary block.

Run from this directory:
    python recover_sim4b_from_log.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "sim4b_run.log"
BROKEN = HERE / "sim4b_mmlu_pro_200.json"
OUT = HERE / "sim4b_mmlu_pro_partial_recovered.json"


HEADER_RE = re.compile(r"^\[(\d+)/200\]\s*\(([^)]+)\)\s*Q:\s*(.*)$")
GT_RE = re.compile(r"^\s*GT:\s*([A-J])\s*$")
BASELINE_RE = re.compile(r"^\s*Baseline:\s*([A-J]|None)\s+(OK|WRONG)\s*$")
PDA_RE = re.compile(r"^\s*PDA:\s*([A-J]|None)\s+(OK|WRONG)\s*$")
WORKERS_RE = re.compile(r"^\s*Workers:\s*(\[[^\]]*\])\s*$")
RUNNING_RE = re.compile(r"^\s*Running:\s*B\s+(\d+)/(\d+).*\|\s*PDA\s+(\d+)/(\d+)")


def parse_log(text: str) -> tuple[list[dict], dict | None]:
    """Line-by-line parser: tolerates multi-line questions.

    Returns (per_item_results, last_running_summary).
    """
    items: list[dict] = []
    cur: dict | None = None
    last_running: dict | None = None

    for raw_line in text.splitlines():
        m = HEADER_RE.match(raw_line)
        if m:
            if cur:
                items.append(cur)
            cur = {
                "index": int(m.group(1)) - 1,
                "category": m.group(2).strip(),
                "question_prefix": m.group(3).strip(),
                "ground_truth": None,
                "baseline_answer": None,
                "baseline_correct": False,
                "pda_answer": None,
                "pda_correct": False,
                "pda_worker_answers": [],
            }
            continue
        if not cur:
            continue
        m = GT_RE.match(raw_line)
        if m:
            cur["ground_truth"] = m.group(1)
            continue
        m = BASELINE_RE.match(raw_line)
        if m:
            cur["baseline_answer"] = m.group(1) if m.group(1) != "None" else None
            cur["baseline_correct"] = m.group(2) == "OK"
            continue
        m = PDA_RE.match(raw_line)
        if m:
            cur["pda_answer"] = m.group(1) if m.group(1) != "None" else None
            cur["pda_correct"] = m.group(2) == "OK"
            continue
        m = WORKERS_RE.match(raw_line)
        if m:
            try:
                cur["pda_worker_answers"] = eval(m.group(1))
            except Exception:
                cur["pda_worker_answers"] = []
            continue
        m = RUNNING_RE.match(raw_line)
        if m:
            last_running = {
                "n_completed": int(m.group(2)),
                "baseline_correct": int(m.group(1)),
                "pda_correct": int(m.group(3)),
            }
            continue
    if cur:
        items.append(cur)
    return items, last_running


def main() -> None:
    if not LOG.exists():
        raise SystemExit(f"Log not found: {LOG}")

    text = LOG.read_text(encoding="utf-8", errors="replace")
    items, last_running = parse_log(text)
    # Drop items that never got their baseline/pda recorded (interrupted last one)
    items = [x for x in items if x["ground_truth"] is not None]
    n = len(items)
    b_correct = sum(1 for x in items if x["baseline_correct"])
    p_correct = sum(1 for x in items if x["pda_correct"])

    # Read original broken summary for config metadata
    broken = json.loads(BROKEN.read_text()) if BROKEN.exists() else {"config": {}}
    cfg = broken.get("config", {})

    # Authoritative cumulative numbers come from the last Running: line.
    # The per-item parser may miss the very-last in-flight item; use the
    # running line for the summary block.
    if last_running:
        n_auth = last_running["n_completed"]
        b_auth = last_running["baseline_correct"]
        p_auth = last_running["pda_correct"]
    else:
        n_auth, b_auth, p_auth = n, b_correct, p_correct

    out = {
        "config": {
            **cfg,
            "recovered_from_log": True,
            "log_file": LOG.name,
            "note": "Original sim4b_mmlu_pro_200.json was written with null answers because the run aborted at item 120/200 (API-cost decision). This file is the per-item data parsed back out of sim4b_run.log. The summary block uses the authoritative cumulative counts from the last 'Running:' line in the log.",
        },
        "summary": {
            "n_completed": n_auth,
            "n_planned": 200,
            "baseline_correct": b_auth,
            "baseline_accuracy_pct": round(100.0 * b_auth / n_auth, 1) if n_auth else 0.0,
            "pda_correct": p_auth,
            "pda_accuracy_pct": round(100.0 * p_auth / n_auth, 1) if n_auth else 0.0,
            "delta_pp": round(100.0 * (p_auth - b_auth) / n_auth, 1) if n_auth else 0.0,
            "per_item_parser_recovered": n,
        },
        "results": items,
    }
    OUT.write_text(json.dumps(out, indent=2))
    s = out["summary"]
    print(f"Recovered {n} items from {LOG.name}")
    print(f"  baseline: {s['baseline_correct']}/{n} = {s['baseline_accuracy_pct']}%")
    print(f"  pda:      {s['pda_correct']}/{n} = {s['pda_accuracy_pct']}%")
    print(f"  delta:    {s['delta_pp']:+.1f}pp")
    print(f"Wrote {OUT.name}")


if __name__ == "__main__":
    main()
