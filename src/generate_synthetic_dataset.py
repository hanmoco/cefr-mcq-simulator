import argparse
from pathlib import Path

import pandas as pd


UNIFORM_DIST = [0.25, 0.25, 0.25, 0.25]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append synthetic learner response distributions used for Settings 1-4. "
            "The resulting file contains the original examples plus all synthetic examples. "
            "Use split_ft_s1.json through split_ft_s4.json to select each setting."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to the processed base JSONL file produced by generate_dataset.py.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Path where the synthetic-augmented JSONL file will be written.",
    )
    return parser.parse_args()


def one_hot(label: int):
    if label not in {0, 1, 2, 3}:
        raise ValueError(f"Unknown label: {label}")
    dist = [0, 0, 0, 0]
    dist[label] = 1
    return dist


def add_new_ids(df_source: pd.DataFrame, start_pid: int) -> pd.DataFrame:
    if df_source.empty:
        return df_source.copy()

    df_new = df_source.copy()
    new_pid_list = []
    new_qid_list = []
    question_number = 1
    first_pid = int(str(df_new.iloc[0]["passage_id"])[1:])

    for i in range(len(df_new)):
        if i == 0:
            prev_pid = None
        else:
            prev_pid = int(str(df_new.iloc[i - 1]["passage_id"])[1:])

        pid = int(str(df_new.iloc[i]["passage_id"])[1:])
        new_pid = start_pid + pid - first_pid
        new_pid_list.append(new_pid)

        if pid == prev_pid:
            question_number += 1
        else:
            question_number = 1

        new_qid_list.append(f"p{new_pid}_q{question_number}")

    df_new["passage_id"] = [f"p{pid}" for pid in new_pid_list]
    df_new["id"] = new_qid_list
    return df_new


def next_start_pid(*dfs: pd.DataFrame) -> int:
    last_non_empty = None
    for df in dfs:
        if not df.empty:
            last_non_empty = df
    if last_non_empty is None:
        raise ValueError("No non-empty dataframe was provided.")
    last_passage_id = str(last_non_empty.iloc[-1]["passage_id"])
    return int(last_passage_id[1:]) + 1


def make_uniform_examples(df: pd.DataFrame, source_levels, target_level: str, start_pid: int) -> pd.DataFrame:
    df_new = df[df["level"].isin(source_levels)].copy()
    df_new["level"] = target_level
    df_new["teacher_soft"] = [UNIFORM_DIST.copy() for _ in range(len(df_new))]
    return add_new_ids(df_new, start_pid)


def make_one_hot_examples(df: pd.DataFrame, source_levels, target_level: str, start_pid: int) -> pd.DataFrame:
    df_new = df[df["level"].isin(source_levels)].copy()
    df_new["level"] = target_level
    df_new["teacher_soft"] = [one_hot(label) for label in df_new["label"].to_list()]
    return add_new_ids(df_new, start_pid)


def build_synthetic_dataset(input_path: Path, output_path: Path) -> int:
    df = pd.read_json(input_path, lines=True)

    # B1 learner, C1/C2 questions: uniform distribution.
    b1_new = make_uniform_examples(
        df=df,
        source_levels=["C1", "C2"],
        target_level="B1",
        start_pid=next_start_pid(df),
    )

    # B2 learner, C2 questions: uniform distribution.
    b2_new = make_uniform_examples(
        df=df,
        source_levels=["C2"],
        target_level="B2",
        start_pid=next_start_pid(b1_new),
    )

    # C1 learner, B1 questions: one-hot correct-answer distribution.
    c1_new = make_one_hot_examples(
        df=df,
        source_levels=["B1"],
        target_level="C1",
        start_pid=next_start_pid(b2_new),
    )

    # C2 learner, B1/B2 questions: one-hot correct-answer distribution.
    c2_new = make_one_hot_examples(
        df=df,
        source_levels=["B1", "B2"],
        target_level="C2",
        start_pid=next_start_pid(c1_new),
    )

    df_out = pd.concat([df, b1_new, b2_new, c1_new, c2_new], ignore_index=True)

    dups = df_out["id"].duplicated().sum()
    if dups > 0:
        raise ValueError(f"Duplicate ids found: {dups}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_json(output_path, orient="records", lines=True, force_ascii=False)
    return len(df_out)


def main() -> None:
    args = parse_args()
    n_examples = build_synthetic_dataset(args.input, args.output)
    print(f"Saved {n_examples} examples to {args.output}")


if __name__ == "__main__":
    main()
