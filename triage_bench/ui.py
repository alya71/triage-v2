"""
Gradio web UI for running triage benchmarks with live progress updates.
"""
import os
import gradio as gr
from benchmark_runner import run_benchmark, TRIAGE_LEVELS

# Model choices organized by provider
OPENAI_MODELS = ["o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-4o", "gpt-4.5-preview"]
DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"]
GROQ_MODELS = ['kimi-k2-instruct-0905', 'allam-2-7b', 'qwen3-32b', 'gpt-oss-20b', 'llama-4-scout-17b-16e-instruct', 'llama-3.1-8b-instant', 'llama-4-maverick-17b-128e-instruct', 'compound', 'llama-3.3-70b-versatile', 'gpt-oss-120b', 'compound-mini', 'kimi-k2-instruct']

PROVIDER_MODELS = {
    "OpenAI": OPENAI_MODELS,
    "DeepSeek": DEEPSEEK_MODELS,
    "Groq": GROQ_MODELS
}


def format_summary(summary):
    """Format summary metrics to match main.py output format."""
    total_preds = summary["total_predictions"]
    overall_acc = summary["overall_accuracy"]
    correct = summary["correct"]
    incorrect = summary["incorrect"]
    per_level_acc = summary["per_level_accuracy"]
    safety_rate = summary["safety_rate"]
    safe_predictions = summary["safe_predictions"]
    overtriage_rate = summary["overtriage_rate"]
    overtriage_errors = summary["overtriage_errors"]
    
    output = "Triage Evaluation Summary (pooled across runs):\n"
    output += f"Total model calls: {total_preds}\n"
    output += f"Overall Accuracy: {overall_acc:.2%}\t({correct} / {total_preds})\n\n"
    
    output += "Accuracy by Triage Level (all predictions):\n"
    for lvl in TRIAGE_LEVELS:
        if lvl in per_level_acc:
            acc_lvl = per_level_acc[lvl]["accuracy"]
            corr_lvl = per_level_acc[lvl]["correct"]
            total_lvl = per_level_acc[lvl]["total"]
            output += f"  {lvl}: {acc_lvl:.2%}\t({corr_lvl}/{total_lvl})\n"
    
    output += f"\nSafety (at‑or‑above correct urgency): {safety_rate:.2%}\t({safe_predictions}/{total_preds})\n"
    output += f"Inclination to Over‑triage (among incorrect): {overtriage_rate:.2%}\t({overtriage_errors}/{incorrect})"
    
    return output


def update_model_choices(provider):
    """Update model dropdown based on selected provider."""
    if provider:
        return gr.update(
            choices=PROVIDER_MODELS[provider],
            value=None,
            interactive=True,
            info=f"Select a {provider} model"
        )
    else:
        return gr.update(
            choices=[],
            value=None,
            interactive=False,
            info="Select a provider first to see available models"
        )


def submit_api_key(key, key_type):
    """Handle API key submission."""
    if key and key.strip():
        return f"✓ {key_type} API key recorded", key.strip(), ""
    return f"Please enter {key_type} API key", "", key


def run_benchmark_ui(model, runs, num_vignettes, key_openai, key_deepseek, key_groq, progress=gr.Progress()):
    """
    Run benchmark with Gradio progress tracking.
    
    Args:
        model: Model name from dropdown
        runs: Number of runs (1-5)
        num_vignettes: Number of vignettes to use (None for all)
        key_openai: OpenAI API key (optional)
        key_deepseek: DeepSeek API key (optional)
        key_groq: Groq API key (optional)
        progress: Gradio progress tracker
    """
    # Convert empty string/None/0 to None for num_vignettes (use all)
    if num_vignettes == "" or num_vignettes is None or num_vignettes == 0:
        num_vignettes = None
    elif num_vignettes > 45:
        return f"Error: Only 45 vignettes exist, but {num_vignettes} were requested.", "Error"
    elif num_vignettes <= 0:
        return "Error: Number of vignettes must be positive or empty for all vignettes.", "Error"
    
    # Set API keys in environment if provided
    if key_openai and key_openai.strip():
        os.environ["KEY_OPENAI"] = key_openai.strip()
    if key_deepseek and key_deepseek.strip():
        os.environ["KEY_DEEPSEEK"] = key_deepseek.strip()
    if key_groq and key_groq.strip():
        os.environ["KEY_GROQ"] = key_groq.strip()
    
    # Reload the specific ummon module that will be used to recreate client with new API key
    # Since ummon modules now read from os.environ directly, we just need to reload the module
    try:
        import importlib
        if model in {"o1", "o1-mini", "o3", "o3-mini", "o4-mini", "gpt-4o", "gpt-4.5-preview"}:
            import medask.ummon.openai
            importlib.reload(medask.ummon.openai)
        elif model in {"deepseek-chat", "deepseek-reasoner"}:
            import medask.ummon.deepseek
            importlib.reload(medask.ummon.deepseek)
        else:
            # Groq models
            import medask.ummon.groq
            importlib.reload(medask.ummon.groq)
    except Exception:
        # If reload fails, continue anyway - the environment variables are set
        pass
    
    try:
        # Create progress callback that updates Gradio progress
        def progress_callback(update_dict):
            completed = update_dict["completed"]
            total = update_dict["total"]
            run = update_dict["run"]
            runs_total = update_dict["runs"]
            current_case = update_dict["current_case"]
            total_cases = update_dict["total_cases"]
            
            # Update progress bar using Gradio's Progress API
            if progress is not None:
                progress(completed / total, desc=f"Run {run}/{runs_total}, Case {current_case}/{total_cases}")
        
        # Run benchmark with progress callback
        summary, results = run_benchmark(
            model=model,
            vignette_set="semigran",
            runs=runs,
            num_vignettes=num_vignettes,
            progress_callback=progress_callback
        )
        
        # Format and return summary
        summary_text = format_summary(summary)
        status_text = f"Benchmark completed! Processed {summary['total_predictions']} predictions."
        return summary_text, status_text
    
    except Exception as e:
        import traceback
        error_msg = f"Error running benchmark: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, "Error occurred"


def create_ui():
    """Create and launch the Gradio interface."""
    with gr.Blocks(title="Triage Benchmark") as demo:
        gr.Markdown("# Medical Triage Benchmark")
        gr.Markdown("Run triage classification benchmarks with live progress tracking.")
        
        # API Keys section at the top
        gr.Markdown("### API Keys")
        gr.Markdown("**Required:** You must either preset API keys as environment variables or input them below to use the corresponding models.")
        
        with gr.Row():
            with gr.Column():
                key_openai_input = gr.Textbox(
                    label="OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    info="Required for OpenAI models"
                )
                key_openai_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False
                )
                key_openai_submit = gr.Button("Submit OpenAI Key", size="sm")
            
            with gr.Column():
                key_deepseek_input = gr.Textbox(
                    label="DeepSeek API Key",
                    type="password",
                    placeholder="...",
                    info="Required for DeepSeek models"
                )
                key_deepseek_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False
                )
                key_deepseek_submit = gr.Button("Submit DeepSeek Key", size="sm")
            
            with gr.Column():
                key_groq_input = gr.Textbox(
                    label="Groq API Key",
                    type="password",
                    placeholder="...",
                    info="Required for Groq models"
                )
                key_groq_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False
                )
                key_groq_submit = gr.Button("Submit Groq Key", size="sm")
        
        # Store submitted keys
        key_openai_stored = gr.State(value="")
        key_deepseek_stored = gr.State(value="")
        key_groq_stored = gr.State(value="")
        
        # API key submit handlers
        def submit_openai_key(key):
            status, stored, clear_input = submit_api_key(key, "OpenAI")
            return status, stored, clear_input
        
        def submit_deepseek_key(key):
            status, stored, clear_input = submit_api_key(key, "DeepSeek")
            return status, stored, clear_input
        
        def submit_groq_key(key):
            status, stored, clear_input = submit_api_key(key, "Groq")
            return status, stored, clear_input
        
        key_openai_submit.click(
            fn=submit_openai_key,
            inputs=[key_openai_input],
            outputs=[key_openai_status, key_openai_stored, key_openai_input]
        )
        key_deepseek_submit.click(
            fn=submit_deepseek_key,
            inputs=[key_deepseek_input],
            outputs=[key_deepseek_status, key_deepseek_stored, key_deepseek_input]
        )
        key_groq_submit.click(
            fn=submit_groq_key,
            inputs=[key_groq_input],
            outputs=[key_groq_status, key_groq_stored, key_groq_input]
        )
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Configuration")
                
                provider_dropdown = gr.Dropdown(
                    choices=["OpenAI", "DeepSeek", "Groq"],
                    value=None,
                    label="Provider",
                    info="Select a provider first"
                )
                
                model_dropdown = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Model",
                    info="Select a provider first to see available models",
                    interactive=False,
                    allow_custom_value=False
                )
                
                runs_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Number of Runs",
                    info="How many stochastic passes per vignette (1-5)"
                )
                
                num_vignettes_input = gr.Number(
                    value=45,
                    label="Number of Vignettes",
                    info="There are 45 vignettes in total",
                    precision=0
                )
                
                run_button = gr.Button("Run Benchmark", variant="primary")
            
            with gr.Column():
                progress_text = gr.Textbox(
                    label="Progress",
                    value="Ready to run benchmark...",
                    interactive=False
                )
                results_output = gr.Textbox(
                    label="Results Summary",
                    value="",
                    lines=15,
                    interactive=False
                )
        
        # Update model dropdown when provider changes
        provider_dropdown.change(
            fn=update_model_choices,
            inputs=[provider_dropdown],
            outputs=[model_dropdown]
        )
        
        def run_with_progress(model, runs, num_vignettes, key_openai_stored, key_deepseek_stored, key_groq_stored, progress=gr.Progress()):
            """Wrapper to run benchmark with progress tracking."""
            if not model:
                return "Error: Please select a model.", "Error: No model selected"
            
            result, status = run_benchmark_ui(
                model, runs, num_vignettes, 
                key_openai_stored, key_deepseek_stored, key_groq_stored,
                progress
            )
            return result, status
        
        run_button.click(
            fn=run_with_progress,
            inputs=[
                model_dropdown, runs_slider, num_vignettes_input,
                key_openai_stored, key_deepseek_stored, key_groq_stored
            ],
            outputs=[results_output, progress_text]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()

