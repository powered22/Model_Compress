python example_main.py \
    --input_jsonl data/weather_extreme/10_dataset.jsonl \
    --output_jsonl output/extreme_weather_zeroshot_qwen_results.jsonl \
    --prompting_mode zeroshot \
    --model qwen2.5:7b-instruct-q2_K \
    --n 10 \
    --logging \
    --concurrency 5 \
    --log_prefix ExtremeMainZeroShot_Qwen10Data \
    --max_log_csv 10 \
    --task weather-extreme \
    > results_log/ExtremeMainZeroShot_Qwen10Data.log 2>&1