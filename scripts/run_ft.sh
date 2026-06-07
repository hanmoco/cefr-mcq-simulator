#!/bin/bash

python src/train.py \
  --data_path data/processed/cmcqrd.jsonl \
  --split_json splits/split_ft.json \
  --out_dir outputs/ft \
  --bf16
