export TOKENIZERS_PARALLELISM=false


python few_shot_selector/cli.py \
    --input_file data/weather/100_test_data.jsonl \
    --output_file output_fewshot_selector/100_test_data_5ClusterSubCluster1_pca.jsonl \
    --stat_file data/100data_5ClusterSubCluster1_pca.json \
    --n_clusters 5 \
    --n_subclusters 1 \
    --viz_method pca \
    --portion 1 \
    --pict_title 100data_5ClusterSubCluster1_pca \
    > results_log/100data_5ClusterSubCluster1_pca.txt 2>&1\


