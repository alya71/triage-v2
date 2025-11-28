"""
Refactored benchmark runner that supports progress callbacks for web UI.
"""
import json
import os
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

from medask.models.comms.models import CMessage
from medask.models.orm.models import Role
from medask.ummon.base import BaseUmmon
from medask.ummon.deepseek import UmmonDeepSeek
from medask.ummon.openai import UmmonOpenAI

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
    - OpenAI models: o1, o1-mini, o3, o3-mini, o4-mini, gpt-4o, gpt-4.5-preview
    - DeepSeek models: deepseek-chat, deepseek-reasoner
    - Groq models: groq+<model-name>
    - Together AI models: together+<model-name>
    """
    # OpenAI models
    if model in {"o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-4o", "gpt-4.5-preview"}:
        return UmmonOpenAI(model)
    
    # Groq models
    elif model.startswith("groq+"):
        from medask.ummon.groq import UmmonGroq
        groq_model = model[len("groq+"):]
        return UmmonGroq(groq_model)
    
    # Together AI models
    elif model.startswith("together+"):
        from medask.ummon.together import UmmonTogether
        together_model = model[len("together+"):]
        return UmmonTogether(together_model)
    
    # DeepSeek models (default)
    else:
        return UmmonDeepSeek(model)


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
        vignettes = [json.loads(l) for l in f]
    
    # Limit vignettes if specified
    if num_vignettes is not None:
        vignettes = vignettes[:num_vignettes]
    
    num_cases = len(vignettes)
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

