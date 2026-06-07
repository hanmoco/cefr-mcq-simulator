import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Cambridge Multiple-Choice Questions Reading Dataset "
            "JSONL file into the JSONL format used by the learner simulator."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to the original Cambridge MCQ Reading Dataset JSONL file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Path where the processed JSONL file will be written.",
    )
    return parser.parse_args()


def normalize_fac_values(fac_values):
    teacher_soft_sum = sum(fac_values)
    if teacher_soft_sum == 0:
        return None
    return [fac / teacher_soft_sum for fac in fac_values]


def answer_to_label(answer: str):
    mapping = {"a": 0, "b": 1, "c": 2, "d": 3}
    return mapping.get(answer.strip().lower())


def convert_dataset(input_path: Path, output_path: Path) -> int:
    df = pd.read_json(input_path, encoding="utf-8", lines=True)
    option_keys = ["a", "b", "c", "d"]
    data = []

    for i in range(len(df)):
        pid = df.iloc[i]["id"]
        passage_id = f"p{pid}"

        title = df.iloc[i]["title"].strip()
        passage = df.iloc[i]["text"].strip()
        text = title + "\n" + passage

        level = df.iloc[i]["level"].strip()
        questions = df.iloc[i]["questions"]

        for n in range(len(questions)):
            q = questions[str(n + 1)]

            question = q["text"].strip()
            answer = q["answer"].strip()
            options_raw = q["options"]

            # Skip questions with missing, None, or empty option text.
            if any(
                key not in options_raw
                or "text" not in options_raw[key]
                or options_raw[key]["text"] is None
                or str(options_raw[key]["text"]).strip() == ""
                for key in option_keys
            ):
                continue

            # Skip questions without learner response ratios.
            if any(
                "fac" not in options_raw[key]
                or options_raw[key]["fac"] is None
                for key in option_keys
            ):
                continue

            options = [options_raw[key]["text"].strip() for key in option_keys]
            fac_values = [options_raw[key]["fac"] for key in option_keys]
            teacher_soft = normalize_fac_values(fac_values)
            if teacher_soft is None:
                continue

            label = answer_to_label(answer)
            if label is None:
                continue

            ex_id = f"{passage_id}_q{n + 1}"
            data.append(
                {
                    "id": ex_id,
                    "passage_id": passage_id,
                    "level": level,
                    "text": text,
                    "question": question,
                    "options": options,
                    "label": label,
                    "teacher_soft": teacher_soft,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return len(data)


def main() -> None:
    args = parse_args()
    n_examples = convert_dataset(args.input, args.output)
    print(f"Saved {n_examples} examples to {args.output}")


if __name__ == "__main__":
    main()
