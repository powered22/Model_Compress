python -m experiments.baselines.train_lstm \
  --task       weather \
  --train-data data/weather/train.jsonl \
  --val-data   data/weather/val.jsonl \
  --seq-len    6 \
  --hmax       10 \
  --hidden-dim 128 \
  --num-layers 2 \
  --epochs     20 \
  --loss       mae \
  --save-dir   results_log

