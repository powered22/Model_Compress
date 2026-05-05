

import argparse
import json
import pandas as pd
from experiments.runner import WeatherExperiment, WeatherExtremeExperiment
from experiments.utils import get_logging
import logging
import datetime
import random

# N = 750
N = 50
TIMEOUT = 180
CONCURRENCY = 50
GEMMA_Q4 = 'gemma_9b_q4:latest'
GEMMA_Q8 = 'gemma_9b_q8:latest'
LLAMA_Q4 = 'llama_8b_q4:latest'
LLAMA_Q8 = 'llama_8b_q8:latest'
QWEN_Q4 = 'qwen_7b_q4:latest'
QWEN_Q8 = 'qwen_7b_q8:latest'


def main(args):
    log_prefix = get_logging(
        name= args.log_prefix,
        log_dir=args.log_dir,
        log_prefix=args.log_prefix,
        rank=args.rank,
        master_only=args.master_only
    )

    log_prefix.info("Starting experiment...")

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        weather_data = [json.loads(line) for line in f]


    if args.task == "weather":
        experiment = WeatherExperiment(
            data_list=weather_data,
            log_prefix=log_prefix,
            prompting_mode=args.prompting_mode,
            model=args.model,
            concurrency=args.concurrency,
            task=args.task,
            format_dbase = args.format_dbase)

        experiment.run(logging=args.logging)

    if args.task == "weather-extreme":
        experiment = WeatherExtremeExperiment(
            data_list=weather_data,
            log_prefix=log_prefix,
            prompting_mode=args.prompting_mode,
            model=args.model,
            concurrency=args.concurrency,
            task=args.task,
            format_dbase = args.format_dbase)

        experiment.run(logging=args.logging)


    log_prefix.info("Experiment Finished...")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run zero- or few-shot weather experiment.")
    parser.add_argument("--input_jsonl", type=str, default="100_random_test_data.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output_jsonl", type=str, default="weather_results.csv", help="Path to output CSV file")
    parser.add_argument("--prompting_mode", type=str, choices=["zeroshot", "fewshot"], default="zeroshot",
                        help="Prompting mode")
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Model name")
    parser.add_argument("--n", type=int, default=10, help="Number of examples to run")
    parser.add_argument("--logging", action="store_true", help="Enable logging of outputs")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrency")
    parser.add_argument('--log_prefix', type=str, default='weather_run', help='Log file prefix')
    parser.add_argument("--max_log_csv", type=int, default=10, help="Number of row to save in csv")
    parser.add_argument('--log_dir', type=str, default='results_log', help='Directory to save log files')
    parser.add_argument('--rank', type=int, default=0, help='Distributed rank (0 for single)')
    parser.add_argument('--master_only', action='store_true', help='Log only from rank 0')
    parser.add_argument("--format_dbase", type=str, default="sentence", help="Format dataset")
    parser.add_argument("--task", type=str, default="weather", help="Task for running experiment")
    args = parser.parse_args()
    main(args)



