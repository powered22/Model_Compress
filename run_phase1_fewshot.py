"""
run_phase1_fewshot.py
Phase 1 launcher: fewshot experiments across all tasks.

Usage:
    python run_phase1_fewshot.py
    python run_phase1_fewshot.py --model mistral:7b-instruct-v0.2-q2_K
    python run_phase1_fewshot.py --tasks weather,extreme
    python run_phase1_fewshot.py --max-samples 10
    python run_phase1_fewshot.py --max-samples 10 --k-folds 3
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request

# ── ANSI colors ───────────────────────────────────────────────────────────────
GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
NC     = "\033[0m"

def info(msg):  print(f"{GREEN}[INFO]{NC}  {msg}")
def warn(msg):  print(f"{YELLOW}[WARN]{NC}  {msg}")
def error(msg): print(f"{RED}[ERROR]{NC} {msg}", file=sys.stderr)


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 — Fewshot experiments")
    parser.add_argument("--model",       default=os.environ.get("MODELS", "llama3:8b-instruct-q2_K"),
                        dest="models",   help="Comma-separated Ollama model tags")
    parser.add_argument("--tasks",       default="weather,extreme,energy",
                        help="Comma-separated tasks to run (weather, extreme, energy)")
    parser.add_argument("--log-dir",     default=os.environ.get("LOG_DIR", "results_log"),
                        dest="log_dir",  help="Output directory for logs")
    parser.add_argument("--max-samples", default=int(os.environ.get("MAX_SAMPLES", 100)),
                        dest="max_samples", type=int,
                        help="Max samples per task (default: 100)")
    parser.add_argument("--k-folds",     default=int(os.environ.get("K_FOLDS", 5)),
                        dest="k_folds",  type=int,
                        help="K-fold CV for extreme weather (default: 5; use 3 for small datasets)")
    parser.add_argument("--n-shots",     default=int(os.environ.get("N_SHOTS", 4)),
                        dest="n_shots",  type=int,
                        help="Few-shot examples for extreme weather (default: 4)")
    parser.add_argument("--seeds",       default=os.environ.get("SEEDS", "42,123,999"),
                        help="Comma-separated seeds for multi-seed evaluation (default: 42,123,999)")
    parser.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                        dest="ollama_host", help="Ollama server URL")
    return parser.parse_args()


# ── Pre-flight checks ─────────────────────────────────────────────────────────
def check_ollama(ollama_host):
    try:
        urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def check_models(ollama_host, models):
    try:
        with urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=3) as resp:
            tags_json = resp.read().decode()
    except Exception:
        return

    for model in models:
        model = model.strip()
        if f'"{model}"' in tags_json:
            info(f"  Model present: {model}")
        else:
            warn(f"  Model NOT found: {model}")
            warn(f"  Pull it with:  ollama pull {model}")
            warn(f"  Continuing — experiment will fail at runtime for this model.")


# ── Task runners ──────────────────────────────────────────────────────────────
def run_task(module, label, env):
    info(f"━━━ TASK: {label} ━━━")
    result = subprocess.run(
        [sys.executable, "-m", module],
        env=env,
    )
    if result.returncode != 0:
        error(f"Task '{label}' exited with code {result.returncode}")
    info(f"━━━ {label} done ━━━\n")


def main():
    args = parse_args()

    models      = args.models
    tasks       = [t.strip() for t in args.tasks.split(",")]
    log_dir     = args.log_dir
    max_samples = args.max_samples
    k_folds     = args.k_folds
    n_shots     = args.n_shots
    seeds       = args.seeds
    ollama_host = args.ollama_host
    modes       = "fewshot"

    info("Phase 1 — Fewshot experiments")
    info(f"  MODELS:      {models}")
    info(f"  MODES:       {modes}")
    info(f"  TASKS:       {', '.join(tasks)}")
    info(f"  OLLAMA_HOST: {ollama_host}")
    info(f"  LOG_DIR:     {log_dir}")
    info(f"  MAX_SAMPLES: {max_samples}")
    info(f"  K_FOLDS:     {k_folds}  (extreme weather CV)")
    info(f"  N_SHOTS:     {n_shots}")
    print()

    if not check_ollama(ollama_host):
        error(f"Ollama is not reachable at {ollama_host}")
        error("Start it with:  ollama serve")
        error("Or with GPU:    CUDA_VISIBLE_DEVICES=0 ollama serve")
        sys.exit(1)
    info(f"Ollama is reachable at {ollama_host}")

    check_models(ollama_host, models.split(","))
    print()

    os.makedirs(log_dir, exist_ok=True)

    # Env vars yang di-pass ke setiap subprocess
    env = os.environ.copy()
    env["MODELS"]      = models
    env["MODES"]       = modes
    env["OLLAMA_HOST"] = ollama_host
    env["LOG_DIR"]     = log_dir
    env["MAX_SAMPLES"] = str(max_samples)
    env["K_FOLDS"]     = str(k_folds)
    env["N_SHOTS"]     = str(n_shots)
    env["SEEDS"]       = seeds

    task_map = {
        "weather": ("experiments.run_all_weather_exp", "weather forecasting (fewshot)"),
        "extreme": ("experiments.run_all_extreme_exp", f"extreme weather detection (fewshot, k-fold={k_folds})"),
        "energy":  ("experiments.run_all_energy_exp",  "energy (forecast + extreme) (fewshot)"),
    }

    start = time.time()
    for task in tasks:
        if task not in task_map:
            warn(f"Unknown task '{task}'. Valid: weather, extreme, energy")
            continue
        module, label = task_map[task]
        run_task(module, label, env)

    elapsed = int(time.time() - start)
    info(f"All tasks finished in {elapsed // 60}m {elapsed % 60}s")
    info(f"Results written to: {log_dir}/")


if __name__ == "__main__":
    main()
