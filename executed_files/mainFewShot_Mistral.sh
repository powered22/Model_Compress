python example_main.py \
    --input_jsonl data/weather/100_test_data.jsonl \
    --output_jsonl output/weather_fewshot_mistral_results.jsonl \
    --prompting_mode fewshot \
    --model mistral:7b-instruct-v0.2-q2_K \
    --n 10 \
    --logging \
    --concurrency 5 \
    --log_prefix mainFewShot_Mistral100Data \
    --max_log_csv 10 \
    > results_log/mainFewShot_Mistral100Data.log 2>&1