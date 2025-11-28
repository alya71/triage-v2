"""
Refactored benchmark runner that supports progress callbacks for web UI.
"""
import json
import os
import random
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
from medask.ummon.base import BaseUmmon


TRIAGE_ORDER = {"sc": 1, "ne": 2, "em": 3}
TRIAGE_LEVELS = ["em", "ne", "sc"]

ACTIVE_PROMPT = """
Please give a triage classification for the following vignette.
The criteria for triage classifications are presented below:
Emergency (em): Call 911 or go directly to the emergency room;
Non-Emergency (ne): Try to see a doctor, but the symptoms do not require immediate attention at an emergency room. For example, you can see a doctor in the next week.
Self-care (sc): Let the health issue get better on its own and review the situation in a few days again.

RESPONSE FORMAT: 
    Provide only one of the following responses:  
    - `em`  
    - `ne`  
    - `sc`

Vignette:
{vignette}
"""


def create_client(model: str) -> BaseUmmon:
    """
    Create appropriate client based on model string.
    
    Supports:
    - OpenAI models: "gpt-4o", "gpt-5.1", "gpt-5-mini", "gpt-5-nano", "o1", "o1-mini", "o3", "o3-mini", "o4-mini"
    - DeepSeek models: deepseek-chat, deepseek-reasoner
    - Groq models: groq+<model-name>
    - Together AI models: together+<model-name>
    """
    if model in {"o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-4o", "gpt-5.1", "gpt-5-mini", "gpt-5-nano"}:
        from medask.ummon.openai import UmmonOpenAI
        return UmmonOpenAI(model)
    
    elif model in {"deepseek-chat", "deepseek-reasoner"}:
        from medask.ummon.deepseek import UmmonDeepSeek
        return UmmonDeepSeek(model)

    else:
        from medask.ummon.groq import UmmonGroq
        try:
            if model == "qwen3-32b":
                model = "qwen/qwen3-32b"
            elif model == "kimi-k2-instruct-0905":
                model = "moonshotai/kimi-k2-instruct-0905"
            elif model == "gpt-oss-20b":
                model = "openai/gpt-oss-20b"
            elif model == "kimi-k2-instruct":
                model = "moonshotai/kimi-k2-instruct"
            elif model == "compound-mini":
                model = "groq/compound-mini"
            elif model == "gpt-oss-120b":
                model = "openai/gpt-oss-120b"
            elif model == "compound":
                model = "groq/compound"
            elif model == "meta-llama/llama-4-maverick-17b-128e-instruct":
                model = 'meta-llama/llama-4-maverick-17b-128e-instruct'
            elif model == "llama-4-scout-17b-16e-instruct":
                model = "meta-llama/llama-4-scout-17b-16e-instruct"
            elif model == "gpt-oss-20b":
                model = "openai/gpt-oss-20b"
            return UmmonGroq(model)
        except Exception as e: 
            GROQ_CHOICES = ['kimi-k2-instruct-0905', 'allam-2-7b', 'qwen3-32b', 'gpt-oss-20b', 'llama-4-scout-17b-16e-instruct', 'llama-3.1-8b-instant', 'llama-4-maverick-17b-128e-instruct', 'compound', 'llama-3.3-70b-versatile', 'gpt-oss-120b', 'compound-mini', 'kimi-k2-instruct']
            raise ValueError(f"Not a valid Groq model: '{model}', valid models with chat completion feature are {GROQ_CHOICES}") from e
    
    

def _llm_triage(client: BaseUmmon, vignette_text: str) -> str:
    """Get triage classification from LLM."""
    prompt = ACTIVE_PROMPT.format(vignette=vignette_text)
    raw = client.inquire(CMessage(user_id=1, body=prompt, role=Role.USER)).body
    cleaned = re.sub(r"[`\s]", " ", raw.lower()).strip()
    match = re.search(r"\b(em|ne|sc)\b", cleaned)
    return match.group(1) if match else cleaned[:50]


def run_benchmark(
    model: str,
    vignette_set: str = "semigran",
    runs: int = 1,
    num_vignettes: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Tuple[Dict, List[Dict]]:
    """
    Run the triage benchmark with progress updates.
    
    Args:
        model: Model name or URL
        vignette_set: Vignette set name (semigran or kopka)
        runs: Number of runs
        num_vignettes: Number of vignettes to use (None for all)
        progress_callback: Optional callback function that receives progress updates
        
    Returns:
        Tuple of (summary_metrics, results_list)
    """
    # Create client
    client = create_client(model)
    
    # Load vignettes
    vignette_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vignettes")
    vignette_fp = os.path.join(vignette_dir, f"{vignette_set}_vignettes.jsonl")
    
    if not os.path.exists(vignette_fp):
        raise FileNotFoundError(f"Vignette file not found: {vignette_fp}")
    
    with open(vignette_fp, encoding="utf-8") as f:
        all_vignettes = [json.loads(l) for l in f]
    
    total_available = len(all_vignettes)
    
    # Validate num_vignettes
    if num_vignettes is not None:
        if num_vignettes > total_available:
            raise ValueError(f"Only {total_available} vignettes exist, but {num_vignettes} were requested.")
        if num_vignettes <= 0:
            raise ValueError("Number of vignettes must be positive or None for all vignettes.")
    
    # If num_vignettes is None, use all vignettes
    if num_vignettes is None:
        num_cases = total_available
    else:
        num_cases = num_vignettes
    
    total_tasks = num_cases * runs
    
    # Initialize counters
    total_pred_counter = Counter()
    per_level_total = Counter()
    per_level_correct = Counter()
    safe_predictions = 0
    overtriage_errors = 0
    results = []
    
    completed = 0
    
    # Run benchmark
    for run in range(1, runs + 1):
        # Sample vignettes for this run (different sample per run if num_vignettes < total)
        if num_vignettes is None:
            vignettes = all_vignettes
        else:
            vignettes = random.sample(all_vignettes, num_vignettes)
        
        for idx, v in enumerate(vignettes, 1):
            # Evaluate single case
            pred = _llm_triage(client, v["case_description"])
            gold = v["urgency_level"].strip().lower()
            correct = pred == gold
            
            rec = {
                "run_id": run,
                "case_id": idx,
                "true_urgency": gold,
                "llm_output": pred,
                "correct": correct,
                "model": model,
            }
            results.append(rec)
            
            # Update counters
            per_level_total[gold] += 1
            if correct:
                per_level_correct[gold] += 1
                total_pred_counter["correct"] += 1
            else:
                total_pred_counter["incorrect"] += 1
            
            # Safety / over-triage
            if pred in TRIAGE_ORDER and gold in TRIAGE_ORDER:
                if TRIAGE_ORDER[pred] >= TRIAGE_ORDER[gold]:
                    safe_predictions += 1
                if (not correct) and TRIAGE_ORDER[pred] > TRIAGE_ORDER[gold]:
                    overtriage_errors += 1
            
            completed += 1
            
            # Call progress callback
            if progress_callback:
                progress_callback({
                    "completed": completed,
                    "total": total_tasks,
                    "run": run,
                    "runs": runs,
                    "current_case": idx,
                    "total_cases": num_cases,
                })
    
    # Calculate summary metrics
    total_preds = sum(total_pred_counter.values())
    overall_acc = total_pred_counter["correct"] / total_preds if total_preds else 0.0
    
    per_level_acc = {}
    for lvl in TRIAGE_LEVELS:
        preds_lvl = per_level_total[lvl]
        corr_lvl = per_level_correct[lvl]
        acc_lvl = corr_lvl / preds_lvl if preds_lvl else 0.0
        per_level_acc[lvl] = {
            "accuracy": acc_lvl,
            "correct": corr_lvl,
            "total": preds_lvl,
        }
    
    safety_rate = safe_predictions / total_preds if total_preds else 0.0
    incorrect_preds = total_pred_counter["incorrect"]
    overtriage_rate = overtriage_errors / incorrect_preds if incorrect_preds else 0.0
    
    summary = {
        "total_predictions": total_preds,
        "overall_accuracy": overall_acc,
        "correct": total_pred_counter["correct"],
        "incorrect": total_pred_counter["incorrect"],
        "per_level_accuracy": per_level_acc,
        "safety_rate": safety_rate,
        "safe_predictions": safe_predictions,
        "overtriage_rate": overtriage_rate,
        "overtriage_errors": overtriage_errors,
        "model": model,
        "vignette_set": vignette_set,
        "runs": runs,
        "num_vignettes": num_cases,
    }
    
    return summary, results

