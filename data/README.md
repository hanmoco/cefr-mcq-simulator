# Data

This directory is used to store raw and processed dataset files.

This repository does not redistribute the Cambridge Multiple-Choice Questions Reading Dataset (CMCQRD). Please obtain the dataset from the [original provider](https://researchdatasets.cambridge.org/datasets/cambridge-multiple-choice-questions-reading-dataset).

Place the raw JSONL file at:

    data/raw/CMCQRD.jsonl

The preprocessing scripts will generate processed files such as:

    data/processed/cmcqrd.jsonl
    data/processed/cmcqrd_synthetic.jsonl

Raw and processed dataset files are excluded from this repository by `.gitignore`.

## Processed data format

The preprocessing script produces a JSONL file. Each line corresponds to one multiple-choice reading comprehension question.

Each example has the following format:

    {
      "id": "p7_q3",
      "passage_id": "p7",
      "level": "B1",
      "text": "passage text",
      "question": "question text",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "label": 2,
      "teacher_soft": [0.12, 0.18, 0.55, 0.15]
    }

Fields:

    id: question ID
    passage_id: passage ID used for train/validation/test splitting
    level: CEFR level of the question
    text: passage text
    question: question text
    options: four answer options in the order of Option A, B, C, and D
    label: zero-based correct option index (0=A, 1=B, 2=C, 3=D)
    teacher_soft: learner response distribution over Option A-D
