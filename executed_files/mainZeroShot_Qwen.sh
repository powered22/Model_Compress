python example_main.py \
    --input_jsonl data/weather/10_test_data.jsonl \
    --output_jsonl output/weather_zeroshot_qwen_results.jsonl \
    --prompting_mode zeroshot \
    --model qwen2.5:7b-instruct-q2_K \
    --n 10 \
    --logging \
    --concurrency 5 \
    --log_prefix mainZeroShot_Qwen100Data \
    --max_log_csv 10 \
    > results_log/mainZeroShot_Qwen10Data.log 2>&1