#!/bin/bash

python3 main.py \
    --epochs 1 \
    --batch_size 10 \
    --learning_rate 0.6 \
    --input_count 784 \
    --hidden_count 80 \
    --output_count 10 \
    --report_path report.csv \
    --config_path config.txt \
    --with_gpu 1
    # --shuffle 0 \