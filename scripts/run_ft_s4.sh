#!/bin/bash

python src/train.py \
  --data_path data/processed/cmcqrd_synthetic.jsonl \
  --split_json splits/split_ft_s4.json \
  --out_dir outputs/ft_s4 \
  --bf16
