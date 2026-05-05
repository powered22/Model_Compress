python example_main.py \
    --input_jsonl data/weather_extreme/10_dataset.jsonl \
    --output_jsonl output/extreme_weather_fewshot_mistral_results.jsonl \
    --prompting_mode fewshot \
    --model mistral:7b-instruct-v0.2-q2_K \
    --n 10 \
    --logging \
    --concurrency 5 \
    --log_prefix ExtremeMainFewShot_Mistral10Data \
    --max_log_csv 10 \
    --task weather-extreme \
    > results_log/ExtremeMainFewShot_Mistral10Data.log 2>&1