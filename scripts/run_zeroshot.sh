#!/bin/bash

python src/zeroshot.py \
  --data_path data/processed/cmcqrd.jsonl \
  --split_json splits/split_ft.json \
  --out_dir outputs/zeroshot
