import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks from output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def merge_llm_synthesis(model, tokenizer, outputs: Dict[str, str], problem: str) -> str:
    """
    Kombiniert mehrere Antworten mit einem weiteren LLM-Call.
    """
    device = model.device
    
    # Baue den Synthese-Prompt
    context = "\n\n".join([f"Perspektive {name}: {text}" for name, text in outputs.items()])
    
    system_prompt = "Du bist ein Synthese-Experte. Dir werden verschiedene Antworten auf ein Problem praesentiert. Erstelle eine optimale, umfassende Antwort, die die besten Aspekte aller Perspektiven vereint."
    user_content = f"Problem: {problem}\n\nAntworten:\n{context}\n\nSynthese: /no_think"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.4,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    return strip_think(tokenizer.decode(response_ids, skip_special_tokens=True))

def merge_logit_average(model, tokenizer, problem: str, perspectives: List[str], max_tokens=100) -> str:
    """
    Mittelt die Logits mehrerer Durchlaeufe (Prompt-PDA).
    """
    device = model.device
    
    # Bereite Inputs vor
    input_texts = []
    for system_prompt in perspectives:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem + " /no_think"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_texts.append(text)
    
    # Tokenize alle Inputs separat (koennten unterschiedliche Laengen haben)
    input_ids_list = [tokenizer(t, return_tensors="pt").input_ids.to(device) for t in input_texts]
    
    generated_tokens = []
    
    for _ in range(max_tokens):
        all_logits = []
        
        # Hol Logits fuer jeden Pfad
        with torch.no_grad():
            for i_ids in input_ids_list:
                outputs = model(i_ids)
                next_token_logits = outputs.logits[:, -1, :]
                all_logits.append(next_token_logits)
        
        # Mittelwert der Logits
        avg_logits = torch.stack(all_logits).mean(dim=0)
        
        # Waehle Token (greedy oder sampling)
        next_token = torch.argmax(avg_logits, dim=-1)
        generated_tokens.append(next_token.item())
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        # Haenge das gewaehlte Token an alle Pfade an
        input_ids_list = [torch.cat([ids, next_token.unsqueeze(0)], dim=-1) for ids in input_ids_list]
        
    return strip_think(tokenizer.decode(generated_tokens, skip_special_tokens=True))

def merge_majority_vote(outputs: Dict[str, str]) -> str:
    """
    Fuer Aufgaben mit kurzen, klaren Antworten (z.B. Multiple Choice oder Zahlen).
    """
    from collections import Counter
    # Bereinigung: oft steht die Antwort am Ende oder in einer Zeile
    clean_outputs = [strip_think(text).strip().split('\n')[-1].lower() for text in outputs.values()]
    counts = Counter(clean_outputs)
    return counts.most_common(1)[0][0]
