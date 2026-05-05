import os
import json
import asyncio
from abc import abstractmethod
from typing import List, Dict, Optional
import re

import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from .utils import OPS, MODES
import experiments.prompts as prompts
from .utils import (Logger, CSV_WEATHER, CSV_EXTREME_WEATHER,
                    extract_weather_json_async, parse_extreme_response)
from .prompts import WeatherPrompting
from .metrics import Metrics
from datetime import datetime, timedelta
import time
import torch

# RAGRetriever is imported lazily inside the experiment classes so that
# projects that don't use RAG-FS don't pay the sentence-transformers load cost.
_RAGRetriever = None

def _get_rag_retriever_class():
    global _RAGRetriever
    if _RAGRetriever is None:
        from .rag_retriever import RAGRetriever
        _RAGRetriever = RAGRetriever
    return _RAGRetriever

curr_dir = os.path.dirname(__file__)

TARGET_FIELDS = [
    "east_west_wind_speed_10m",
    "north_south_wind_speed_10m",
    "dewpoint_temperature_2m",
    "air_temperature_2m",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
]

# ─────────────────────────────────────────────────────────────────────────────
# Per-model default configuration
# Override timeout and concurrency per model here.
# q2_K models are slower and need more time; reduce concurrency to avoid
# starving parallel calls when the model is memory-constrained.
# ─────────────────────────────────────────────────────────────────────────────

_ON_GPU = torch.cuda.is_available()

MODEL_DEFAULTS = {
    # ── Llama 3 8B Instruct ───────────────────────────────────────────────────
    # q2_K  : most compressed, fastest, lowest quality  (existing baseline)
    "llama3:8b-instruct-q2_K": {
        "concurrency": 2,
        "timeout":     600,
    },
    # q4_K_M: recommended balance between speed and quality
    "llama3:8b-instruct-q4_K_M": {
        "concurrency": 2,
        "timeout":     700,
    },
    # q8_0  : near-lossless, much slower and heavier
    "llama3:8b-instruct-q8_0": {
        "concurrency": 1,
        "timeout":     900,
    },
    # f16   : full precision (largest memory footprint)
    "llama3:8b-instruct-f16": {
        "concurrency": 1,
        "timeout":     1200,
    },

    # ── Mistral 7B Instruct v0.2 ──────────────────────────────────────────────
    "mistral:7b-instruct-v0.2-q2_K": {
        "concurrency": 2 if _ON_GPU else 5,
        "timeout":     600,
    },
    "mistral:7b-instruct-v0.2-q4_K_M": {
        "concurrency": 2,
        "timeout":     700,
    },
    "mistral:7b-instruct-v0.2-q8_0": {
        "concurrency": 1,
        "timeout":     900,
    },
    "mistral:7b-instruct-v0.2-f16": {
        "concurrency": 1,
        "timeout":     1200,
    },

    # ── Qwen 2.5 7B Instruct ─────────────────────────────────────────────────
    "qwen2.5:7b-instruct-q2_K": {
        "concurrency": 2 if _ON_GPU else 5,
        "timeout":     600,
    },
    "qwen2.5:7b-instruct-q4_K_M": {
        "concurrency": 2,
        "timeout":     700,
    },
    "qwen2.5:7b-instruct-q8_0": {
        "concurrency": 1,
        "timeout":     900,
    },
    "qwen2.5:7b-instruct-f16": {
        "concurrency": 1,
        "timeout":     1200,
    },
}

# Global fallback if model not in MODEL_DEFAULTS
_DEFAULT_CONCURRENCY = 2
_DEFAULT_TIMEOUT     = 600


def get_model_config(model: str, concurrency: int = None, timeout: int = None) -> Dict:
    """
    Return effective (concurrency, timeout) for a model.
    Explicit arguments override MODEL_DEFAULTS, which override global defaults.
    """
    defaults = MODEL_DEFAULTS.get(model, {})
    return {
        "concurrency": concurrency or defaults.get("concurrency", _DEFAULT_CONCURRENCY),
        "timeout":     timeout     or defaults.get("timeout",     _DEFAULT_TIMEOUT),
    }


def get_headers(subject: str) -> List[str]:
    headers = json.load(open(os.path.join(curr_dir, '../data/headers.json')))[subject]
    return headers


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_horizon(question: str) -> int:
    import re
    m = re.search(r"\bnext\s+(\d+)\s+hours?\b", question.lower())
    if not m:
        raise ValueError(f"Cannot parse horizon from: {question}")
    return int(m.group(1))


def _persistence_predict(obs_json: dict, horizon: int) -> dict:
    obs_timestamps = sorted([k for k in obs_json.keys() if k != "question"])
    last_ts        = obs_timestamps[-1]
    last_row       = obs_json[last_ts]
    last_values    = {f: last_row.get(f) for f in TARGET_FIELDS}
    last_dt        = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
    prediction     = {}
    for h in range(1, horizon + 1):
        future_ts              = (last_dt + timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")
        prediction[future_ts]  = dict(last_values)
    return prediction


# ─────────────────────────────────────────────────────────────────────────────
# Base Experiment
# ─────────────────────────────────────────────────────────────────────────────

class Experiment:
    def __init__(self,
                 data,
                 prompting: prompts.Prompting,
                 model: str = "llama3",
                 op: OPS = None,
                 mode: MODES = None,
                 concurrency: int = None,
                 timeout: int = None):
        self.data      = data
        self.model     = model
        self.metrics   = Metrics()
        self.prompting = prompting
        self.op        = op
        self.mode      = mode
        self.rp        = prompts.ResponseProcessor()

        # Resolve effective config — explicit args override model defaults
        cfg = get_model_config(model, concurrency, timeout)
        self.concurrency = cfg["concurrency"]
        self.timeout     = cfg["timeout"]
        self.semaphore   = asyncio.Semaphore(self.concurrency)

    @abstractmethod
    def run(self, logging: bool = False):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Persistence Baseline
# ─────────────────────────────────────────────────────────────────────────────

class PersistenceExperiment:
    def __init__(self, data_list: list, log_prefix):
        self.data       = {str(i): d for i, d in enumerate(data_list)}
        self.log_prefix = log_prefix
        self.metrics    = Metrics()
        self.results    = []

    def run(self, logging: bool = True) -> dict:
        logger = Logger("weather", "persistence", CSV_WEATHER) if logging else None

        for key, datum in self.data.items():
            try:
                obs_json     = datum["obs_json"]
                ground_truth = datum["ground_truth"]
                question     = obs_json.get("question", "")
                horizon      = _parse_horizon(question)
                pred         = _persistence_predict(obs_json, horizon)

                scores = self.metrics.update_weather(
                    json_llm_response=pred,
                    sentence_llm_response="",
                    ground_truth=ground_truth,
                    result_sentence="",
                    mode_json=True,
                )
                self.results.append({"index": key, "horizon": horizon,
                                     "ground_truth": ground_truth, **scores})

                if logger:
                    logger.log([
                        key,
                        f"persistence(h={horizon})",
                        ground_truth,
                        json.dumps(pred),
                        json.dumps(pred),
                        scores['levenshtein_distance'],
                        scores['norm_levenshtein_distance'],
                        scores['lcs'],
                        scores['norm_lcs'],
                        scores['l1loss'],  # penalised MAE
                        scores['l1loss_matched'],  # matched MAE
                        scores['timestamp_coverage'],  # always 100% for persistence
                        scores['coverage'],
                        scores['bertscore'],
                        scores['cosinesim'],
                        scores['bleu'],
                        scores['rouge_l'],
                        scores['strict'],
                        scores['lenient'],
                        "persistence", None, None, "persistence", 0.0,
                    ])

            except Exception as e:
                self.log_prefix.info(f"[Persistence ERROR] key={key}: {e}")

        final_scores = self.metrics.get(subject="weather")
        if logger:
            logger.end()
        self.log_prefix.info("\n=== Persistence Baseline ===")
        for k, v in final_scores.items():
            self.log_prefix.info(f"  {k}: {v}")
        return final_scores


def compute_skill_scores(llm_scores: dict, persistence_scores: dict) -> dict:
    skill = {}
    for key in ["l1loss", "coverage"]:
        p = persistence_scores.get(key, 0)
        l = llm_scores.get(key, 0)
        skill[f"skill_{key}"] = round(1 - (l / p), 4) if p > 0 else None
    return skill


# ─────────────────────────────────────────────────────────────────────────────
# Weather Experiment
# ─────────────────────────────────────────────────────────────────────────────

class WeatherExperiment(Experiment):
    def __init__(self,
                 data_list,
                 log_prefix,
                 prompting_mode,
                 model,
                 task,
                 format_dbase,
                 concurrency: int = None,
                 timeout: int = None,
                 persistence_scores: dict = None,
                 rag_retriever=None,   # RAGRetriever instance for ragfs mode
                 n_shots: int = 4):   # how many examples to retrieve per query

        data_dict      = {str(i): d for i, d in enumerate(data_list)}
        prompting      = WeatherPrompting(prompting=prompting_mode,
                                          task=task, format_dbase=format_dbase)
        self.results            = []
        self.format_dbase       = format_dbase
        self.log_prefix         = log_prefix
        self.task               = task
        self.persistence_scores = persistence_scores
        self.rag_retriever      = rag_retriever
        self.n_shots            = n_shots

        super().__init__(data=data_dict, prompting=prompting, model=model,
                         op=None, mode=None,
                         concurrency=concurrency, timeout=timeout)

    def run(self, logging: bool = True):
        self.results = []
        logger       = Logger(self.task, self.model, CSV_WEATHER) if logging else None
        self.log_prefix.info(
            f"[WeatherExperiment] model={self.model} "
            f"concurrency={self.concurrency} timeout={self.timeout}s"
        )

        async def process_example(key, datum):
            async with self.semaphore:
                try:
                    prompt = self.prompting.get_prompt(datum)

                    # RAG-FS: build a per-datum fewshot prompt dynamically
                    if self.rag_retriever is not None and self.rag_retriever.is_fitted():
                        from .kfold_weather import format_ragfs_weather_prompt
                        retrieved, _sims = self.rag_retriever.retrieve(
                            datum, k=self.n_shots)
                        fewshot = format_ragfs_weather_prompt(retrieved)
                    else:
                        fewshot = self.prompting.get_initial_prompt()

                    start = time.perf_counter()

                    llm_response = await asyncio.wait_for(
                        prompts.talk_to_llm(prompt,
                                            fewshot=fewshot,
                                            model=self.model),
                        timeout=self.timeout)
                    elapsed = time.perf_counter() - start

                    try:
                        json_llm_response = await extract_weather_json_async(llm_response)
                    except Exception:
                        json_llm_response = {}

                    sentence_llm_response = self.rp.process_response(
                        self.task, llm_response, datum["result"])
                    ground_truth   = datum["ground_truth"]
                    result_sentence = datum["result"]
                    mode_json      = self.format_dbase == "jsonl"

                    scores = self.metrics.update_weather(
                        json_llm_response, sentence_llm_response,
                        ground_truth, result_sentence, mode_json)

                    self.results.append({"index": key, "prompt": prompt,
                                         "response": llm_response,
                                         "ground_truth": ground_truth, **scores})

                    if logger:
                        logger.log([
                            key,
                            prompt,
                            ground_truth,
                            llm_response,
                            json_llm_response,
                            scores['levenshtein_distance'],
                            scores['norm_levenshtein_distance'],
                            scores['lcs'],
                            scores['norm_lcs'],
                            scores['l1loss'],  # penalised MAE
                            scores['l1loss_matched'],  # matched MAE (may be None)
                            scores['timestamp_coverage'],  # % ts matched
                            scores['coverage'],
                            scores['bertscore'],
                            scores['cosinesim'],
                            scores['bleu'],
                            scores['rouge_l'],
                            scores['strict'],
                            scores['lenient'],
                            self.model,
                            self.op,
                            self.mode,
                            self.prompting.prompting,
                            elapsed,
                        ])

                except asyncio.TimeoutError:
                    self.log_prefix.info(
                        f"[TIMEOUT] key={key} model={self.model} "
                        f"after {self.timeout}s — sample skipped")
                except Exception as e:
                    self.log_prefix.info(f"[ERROR] key={key}: {e}")

        async def run_tasks():
            tasks = [process_example(k, d) for k, d in self.data.items()]
            await tqdm_asyncio.gather(*tasks, desc="Processing Weather")

        asyncio.run(run_tasks())

        # Warn if fewer samples processed than expected
        n_expected  = len(self.data)
        n_processed = len(self.results)
        if n_processed < n_expected:
            self.log_prefix.info(
                f"[WARNING] Only {n_processed}/{n_expected} samples processed "
                f"for {self.model}. Check for timeouts above.")

        final_scores = self.metrics.get(subject=self.task)

        if self.persistence_scores:
            skill = compute_skill_scores(final_scores, self.persistence_scores)
            final_scores.update(skill)

        if logger:
            logger.log([
                'Final Avg.', None, None, None, None,
                final_scores['levenshtein_distance'],
                final_scores['norm_levenshtein_distance'],
                final_scores['lcs'],
                final_scores['norm_lcs'],
                final_scores['l1loss'],  # penalised MAE
                final_scores['l1loss_matched'],  # matched MAE
                final_scores['timestamp_coverage'],
                final_scores['coverage'],
                final_scores['bertscore'],
                final_scores['cosinesim'],
                final_scores['bleu'],
                final_scores['rouge_l'],
                final_scores['strict'],
                final_scores['lenient'],
                self.model, self.op, self.mode, self.prompting.prompting,
            ])
            logger.end()

        self.log_prefix.info(f"Model={self.model} | Prompting={self.prompting.prompting}")
        for metric, value in final_scores.items():
            self.log_prefix.info(f"  {metric}: {value}")
        return final_scores


# ─────────────────────────────────────────────────────────────────────────────
# Extreme Weather Experiment
# ─────────────────────────────────────────────────────────────────────────────

class WeatherExtremeExperiment(Experiment):
    def __init__(self,
                 data_list,
                 log_prefix,
                 prompting_mode,
                 model,
                 task,
                 format_dbase,
                 concurrency: int = None,
                 timeout: int = None,
                 rag_retriever=None,   # RAGRetriever instance for ragfs mode
                 n_shots: int = 4):   # how many examples to retrieve per query

        data_dict  = {str(i): d for i, d in enumerate(data_list)}
        prompting  = WeatherPrompting(prompting=prompting_mode,
                                      task=task, format_dbase=format_dbase)
        self.task           = task
        self.format_dbase   = format_dbase
        self.results        = []
        self.log_prefix     = log_prefix
        self.rag_retriever  = rag_retriever
        self.n_shots        = n_shots

        super().__init__(data=data_dict, prompting=prompting, model=model,
                         op=None, mode=None,
                         concurrency=concurrency, timeout=timeout)

    def run(self, logging: bool = True):
        logger = Logger(self.task, self.model, CSV_EXTREME_WEATHER) if logging else None
        self.log_prefix.info(
            f"[WeatherExtremeExperiment] model={self.model} "
            f"concurrency={self.concurrency} timeout={self.timeout}s"
        )

        async def process_example(key, datum):
            async with self.semaphore:
                try:
                    prompt = self.prompting.get_prompt(datum)

                    # RAG-FS: build a per-datum fewshot prompt dynamically
                    if self.rag_retriever is not None and self.rag_retriever.is_fitted():
                        from .kfold_extreme import format_ragfs_extreme_prompt
                        retrieved, _sims = self.rag_retriever.retrieve(
                            datum, k=self.n_shots)
                        fewshot = format_ragfs_extreme_prompt(retrieved)
                    else:
                        fewshot = self.prompting.get_initial_prompt()

                    start = time.perf_counter()

                    response = await asyncio.wait_for(
                        prompts.talk_to_llm(
                            prompt,
                            fewshot=fewshot,
                            model=self.model),
                        timeout=self.timeout)
                    elapsed = time.perf_counter() - start

                    text = response.strip().lower()
                    if text.startswith("yes"):
                        pred_has = True
                    elif text.startswith("no"):
                        pred_has = False
                    else:
                        m = re.search(r"\b(yes|no)\b", text)
                        pred_has = (m.group(1) == "yes") if m else False

                    norm = re.sub(r"\s+", " ",
                                  text.replace("-", " ").replace("_", " "))
                    allowed_types = ["hail", "thunderstorm wind", "flash flood",
                                     "tornado", "lightning", "flood", "funnel cloud"]
                    pred_type = "NA"
                    for t in allowed_types:
                        if (t in norm) or (t.replace(" ", "") in norm.replace(" ", "")):
                            pred_type = t.title()
                            break

                    gt_obj   = datum["ground_truth"]
                    _, gt    = next(iter(gt_obj.items()))
                    gt_has   = bool(gt["has_extreme_weather"])
                    gt_type  = str(gt["event_type"])

                    scores = self.metrics.update_weather_extreme(
                        pred_has=pred_has, pred_type=pred_type,
                        gt_has=gt_has,    gt_type=gt_type)

                    self.results.append({
                        "index": key, "response": response,
                        "pred_has": pred_has, "pred_type": pred_type,
                        "gt_has": gt_has,     "gt_type": gt_type,
                        **scores
                    })

                    if logger:
                        logger.log([key, prompt, response, pred_has, pred_type,
                                    gt_has, gt_type,
                                    scores["strict"], scores["lenient"],
                                    scores["consistent"],
                                    self.model, self.prompting.prompting, elapsed])

                except asyncio.TimeoutError:
                    # Explicit timeout log — makes skipped samples visible
                    self.log_prefix.info(
                        f"[TIMEOUT] key={key} model={self.model} "
                        f"after {self.timeout}s — sample skipped")
                except Exception as e:
                    self.log_prefix.info(f"[ERROR] key={key}: {e}")

        async def run_tasks():
            tasks = [process_example(k, d) for k, d in self.data.items()]
            await tqdm_asyncio.gather(*tasks, desc="Processing Extreme Weather")

        asyncio.run(run_tasks())

        # Warn explicitly if samples were skipped
        n_expected  = len(self.data)
        n_processed = len(self.results)
        if n_processed < n_expected:
            self.log_prefix.info(
                f"[WARNING] Only {n_processed}/{n_expected} samples processed "
                f"for {self.model}. Check [TIMEOUT] lines above.")

        final_scores = self.metrics.get(subject=self.task)
        self.log_prefix.info(f"\nModel={self.model}")
        for k, v in final_scores.items():
            self.log_prefix.info(f"  {k}: {v}")

        if logger:
            logger.log(["Final Avg.", None, None,
                        f"TP:{final_scores['tp']}",
                        f"FP:{final_scores['fp']}",
                        f"FN:{final_scores['fn']}",
                        f"TN:{final_scores['tn']}",
                        final_scores["strict"],
                        final_scores["lenient"],
                        final_scores["consistency"],
                        self.model, self.prompting.prompting])
            logger.end()

        return final_scores