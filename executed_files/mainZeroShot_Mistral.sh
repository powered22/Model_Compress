python example_main.py \
    --input_jsonl data/weather/10_test_data.jsonl \
    --output_jsonl output/weather_zeroshot_mistral_results.jsonl \
    --prompting_mode zeroshot \
    --model mistral:7b-instruct-v0.2-q2_K \
    --n 10 \
    --logging \
    --concurrency 5 \
    --log_prefix mainZeroShot_Mistral10Data \
    --max_log_csv 10 \
    > results_log/mainZeroShot_Mistral10Data.log 2>&1