# Data

This repository does not redistribute the Cambridge Multiple-Choice Questions Reading Dataset (CMCQRD).

CMCQRD requires a separate data access request. Please obtain the dataset from the original provider and place the raw JSONL file at:

    data/raw/CMCQRD.jsonl

Raw and processed dataset files are excluded from this repository.

## Preprocessing

After obtaining the raw dataset, run:

    python src/generate_dataset.py \
      --input data/raw/CMCQRD.jsonl \
      --output data/processed/cmcqrd.jsonl

To generate the synthetic response distributions used for Settings 1--4, run:

    python src/generate_synthetic_dataset.py \
      --input data/processed/cmcqrd.jsonl \
      --output data/processed/cmcqrd_synthetic.jsonl

## Processed data format

The preprocessing script produces a JSONL file. Each line corresponds to one multiple-choice reading comprehension question.

Each example contains fields such as:

    {
      "qid": "question id",
      "pid": "passage id",
      "level": "B1",
      "passage": "passage text",
      "question": "question text",
      "options": ["option A", "option B", "option C", "option D"],
      "answer": "A",
      "teacher_soft": [0.25, 0.25, 0.25, 0.25]
    }

The exact field names may depend on the raw CMCQRD format. See src/generate_dataset.py for details.
