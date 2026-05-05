python example_main.py \
    --input_jsonl data/weather/100_test_data.jsonl \
    --output_jsonl output/weather_fewshot_qwen_results.jsonl \
    --prompting_mode fewshot \
    --model qwen2.5:7b-instruct-q2_K \
    --n 10 \
    --logging \
    --concurrency 5 \
    --log_prefix mainFewShot_qwen100Data \
    --max_log_csv 10 \
    > results_log/mainFewShot_qwen100Data.log 2>&1