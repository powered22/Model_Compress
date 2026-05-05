python example_main.py \
    --input_jsonl data/weather_extreme/10_dataset.jsonl \
    --output_jsonl output/extreme_weather_zeroshot_llama_results.jsonl \
    --prompting_mode zeroshot \
    --model llama3:8b-instruct-q2_K \
    --n 10 \
    --logging \
    --concurrency 5 \
    --log_prefix ExtremeMainZeroShot_Llama10Data \
    --max_log_csv 10 \
    --task weather-extreme \
    > results_log/ExtremeMainZeroShot_Llama10Data.log 2>&1