#!/bin/bash

python3 main.py \
    --epochs 1 \
    --batch_size 8000 \
    --learning_rate 0.9 \
    --input_count 784 \
    --hidden_count 10 \
    --output_count 10 \
    --report_path report.csv \
    --config_path config.txt \
    --shuffle 0