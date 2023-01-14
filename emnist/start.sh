#!/bin/bash

python3 main.py \
    --epochs 20 \
    --batch_size 80000 \
    --learning_rate 0.6 \
    --shuffle 1 \
    --input_count 784 \
    --hidden_count 80 \
    --output_count 10 \
    --report_path report.csv \
    --config_path config.txt