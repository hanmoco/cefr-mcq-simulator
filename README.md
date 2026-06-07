# CEFR MCQ Learner Simulator

This repository contains code for simulating CEFR-level learner response distributions for multiple-choice English reading comprehension questions.

The method fine-tunes a large language model to predict the option selection distribution of learners at different CEFR levels. It also supports synthetic response distributions for additional training settings.

## Repository structure

    src/
      generate_dataset.py
      generate_synthetic_dataset.py
      train.py
      zeroshot.py

    splits/
      split_ft.json
      split_ft_s1.json
      split_ft_s2.json
      split_ft_s3.json
      split_ft_s4.json

    data/
      README.md

    scripts/
      run_zeroshot.sh
      run_ft.sh
      run_ft_s1.sh
      run_ft_s2.sh
      run_ft_s3.sh
      run_ft_s4.sh

## Dataset

This repository does not redistribute the Cambridge Multiple-Choice Questions Reading Dataset (CMCQRD).

Please obtain the dataset from the original provider and place the raw JSONL file at:

    data/raw/CMCQRD.jsonl

See data/README.md for details.

## Installation

Install the required Python packages with:

    pip install -r requirements.txt

## Preprocessing

Generate the processed dataset with:

    python src/generate_dataset.py \
      --input data/raw/CMCQRD.jsonl \
      --output data/processed/cmcqrd.jsonl

Generate the synthetic dataset used for Settings 1--4 with:

    python src/generate_synthetic_dataset.py \
      --input data/processed/cmcqrd.jsonl \
      --output data/processed/cmcqrd_synthetic.jsonl

## Training

Fine-tune the model using the original training split:

    bash scripts/run_ft.sh

Fine-tune the model using synthetic Setting 1--4 splits:

    bash scripts/run_ft_s1.sh
    bash scripts/run_ft_s2.sh
    bash scripts/run_ft_s3.sh
    bash scripts/run_ft_s4.sh

## Zero-shot evaluation

Run zero-shot evaluation with:

    bash scripts/run_zeroshot.sh

## Splits

The split files are passage-level train/validation/test splits.

    split_ft.json: original fine-tuning setting
    split_ft_s1.json: Setting 1
    split_ft_s2.json: Setting 2
    split_ft_s3.json: Setting 3
    split_ft_s4.json: Setting 4

## Notes

Raw data, processed data, model checkpoints, LoRA adapters, and output files are excluded from this repository.

## Citation

If you use this code, please cite the corresponding paper.

