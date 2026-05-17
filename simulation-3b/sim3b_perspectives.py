import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List

PERSPECTIVES = {
    "analytisch": "Zerlege das Problem in Teilschritte. Arbeite systematisch und logisch. Beantworte die Frage praezise.",
    "intuitiv": "Erklaere es so, wie du es einem Kind erklaeren wuerdest. Nutze einfache Sprache und klare Bilder.",
    "kritisch": "Hinterfrage die gaengige Erklaerung. Was koennte falsch sein oder welche Einschraenkungen gibt es?",
    "kreativ": "Finde eine ungewoehnliche Analogie oder Metapher, um das Problem zu beschreiben.",
    "praktisch": "Konzentriere dich auf das, was man beobachten, testen und direkt anwenden kann."
}

def generate_perspectives(model, tokenizer, problem: str, perspectives: Dict[str, str] = PERSPECTIVES, max_tokens: int = 200) -> Dict[str, str]:
    """
    Generiert Antworten fuer ein Problem aus verschiedenen Perspektiven.
    """
    results = {}
    device = model.device
    
    for name, system_prompt in perspectives.items():
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem + " /no_think"}
        ]
        
        # Qwen3 uses chat templates
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Generation settings from plan: Thinking Mode recommended T=0.6, top_p=0.95
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract only the generated part
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Strip any remaining think tags
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        results[name] = response
        
    return results

if __name__ == "__main__":
    # Test block (minimal)
    model_id = "Qwen/Qwen3-0.6B"
    print(f"Loading model {model_id} for test...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cpu")
    
    problem = "Warum ist der Himmel blau?"
    outputs = generate_perspectives(model, tokenizer, problem, {"analytisch": PERSPECTIVES["analytisch"]})
    print(f"Result (analytisch): {outputs['analytisch'][:100]}...")
