python example_main.py \
    --input_jsonl data/weather_extreme/10_dataset.jsonl \
    --output_jsonl output/extreme_weather_fewshot_qwen_results.jsonl \
    --prompting_mode fewshot \
    --model qwen2.5:7b-instruct-q2_K \
    --n 10 \
    --logging \
    --concurrency 5 \
    --log_prefix ExtremeMainFewShot_Qwen10Data \
    --max_log_csv 10 \
    --task weather-extreme \
    > results_log/ExtremeMainFewShot_Qwen10Data.log 2>&1