import torch
import sacrebleu
from typing import Dict, List, Optional
import numpy as np

def compute_diversity(outputs: Dict[str, str]) -> float:
    """
    Berechnet die paarweise BLEU-Distanz (1 - BLEU) zwischen den Antworten.
    """
    names = list(outputs.keys())
    texts = list(outputs.values())
    
    if len(texts) < 2:
        return 0.0
    
    bleu_scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            # sacrebleu expects list of references (each reference is a list of strings)
            score = sacrebleu.sentence_bleu(texts[i], [texts[j]]).score
            bleu_scores.append(score / 100.0) # normalize to 0-1
            
    avg_bleu = np.mean(bleu_scores)
    return 1.0 - avg_bleu

def compute_quality_heuristic(problem: str, answer: str, reference: Optional[str] = None) -> Dict[str, float]:
    """
    Heuristic quality scoring (works without a judge model).
    Returns multiple sub-scores.
    """
    import re

    # Strip think tags
    answer_clean = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()

    scores = {}

    # 1. Length score: longer answers are generally more informative (up to a point)
    word_count = len(answer_clean.split())
    scores["length"] = min(word_count / 100, 1.0)  # saturates at 100 words

    # 2. Repetition penalty: detect repeated phrases
    words = answer_clean.lower().split()
    if len(words) > 5:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 1.0
        scores["non_repetition"] = unique_ratio
    else:
        scores["non_repetition"] = 0.5

    # 3. Keyword overlap with problem (relevance)
    problem_words = set(problem.lower().split())
    answer_words = set(answer_clean.lower().split())
    # Remove very common words
    stopwords = {"der", "die", "das", "ist", "ein", "eine", "und", "oder", "in", "von",
                 "the", "is", "a", "an", "and", "or", "in", "of", "to", "for", "it", "that", "this"}
    problem_keywords = problem_words - stopwords
    if problem_keywords:
        overlap = len(problem_keywords & answer_words) / len(problem_keywords)
        scores["relevance"] = overlap
    else:
        scores["relevance"] = 0.5

    # 4. Structure: has paragraphs, lists, or multiple sentences
    sentence_count = len(re.split(r'[.!?]+', answer_clean))
    scores["structure"] = min(sentence_count / 5, 1.0)

    # 5. If reference provided: BLEU-like overlap
    if reference:
        ref_words = set(reference.lower().split()) - stopwords
        ans_words = set(answer_clean.lower().split()) - stopwords
        if ref_words:
            scores["ref_overlap"] = len(ref_words & ans_words) / len(ref_words)

    # Combined score
    weights = {"length": 0.2, "non_repetition": 0.3, "relevance": 0.2, "structure": 0.15, "ref_overlap": 0.15}
    total = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        if key in scores:
            total += scores[key] * weight
            weight_sum += weight
    scores["combined"] = total / weight_sum if weight_sum > 0 else 0.5

    return scores


def compute_quality_llm(model, tokenizer, problem: str, answer: str, reference: Optional[str] = None) -> float:
    """
    Quality score. Uses heuristic scoring (LLM-as-Judge doesn't work with 0.6B).
    Returns combined score 0-1.
    """
    scores = compute_quality_heuristic(problem, answer, reference)
    return scores["combined"]

def compare_with_baselines(model, tokenizer, problem: str, n_perspectives=3) -> Dict:
    """
    Vergleicht PDA mit Single Pass Baseline.
    """
    from sim3b_perspectives import generate_perspectives, PERSPECTIVES
    from sim3b_merge import merge_llm_synthesis
    
    # 1. Single Pass
    results = {}
    import re
    single_pass_output = generate_perspectives(model, tokenizer, problem, {"default": "Beantworte die Frage praezise."})["default"]
    single_pass_output = re.sub(r'<think>.*?</think>', '', single_pass_output, flags=re.DOTALL).strip()
    results["single_pass"] = single_pass_output
    
    # 2. PDA
    perspective_subset = {k: PERSPECTIVES[k] for k in list(PERSPECTIVES.keys())[:n_perspectives]}
    pda_outputs = generate_perspectives(model, tokenizer, problem, perspective_subset)
    pda_merged = merge_llm_synthesis(model, tokenizer, pda_outputs, problem)
    results["pda_merged"] = pda_merged
    
    # 3. Scores (heuristic)
    scores_single = compute_quality_heuristic(problem, single_pass_output)
    scores_pda = compute_quality_heuristic(problem, pda_merged)
    results["score_single"] = scores_single["combined"]
    results["score_pda"] = scores_pda["combined"]
    results["scores_single_detail"] = scores_single
    results["scores_pda_detail"] = scores_pda

    return results
