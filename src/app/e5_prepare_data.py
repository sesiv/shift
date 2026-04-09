"""Prepare train/validation/test splits and IDF artifacts for E5 experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from consts import EMBEDDING_MODEL, MAX_VECTOR_LENGTH
from e5_experiment_data import (
    build_pairs,
    build_triplets,
    compute_idf_dictionary,
    load_ticket_records,
    save_idf_dictionary,
    save_jsonl,
    save_records,
    split_records,
    summarize_records,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset splits, positive/negative pairs, and token-id IDF for E5 pooling experiments."
    )
    parser.add_argument(
        "--source",
        default="src/data/ExportSDLab.xlsx",
        help="Path to the source XLSX file with ticket data.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/e5_pooling",
        help="Directory where prepared artifacts will be saved.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--max-positive-pairs-per-class", type=int, default=200)
    parser.add_argument("--negative-pairs-per-positive", type=int, default=1)
    parser.add_argument("--negatives-per-anchor", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_ticket_records(arguments.source)
    split_map = split_records(
        records,
        seed=arguments.seed,
        val_ratio=arguments.val_ratio,
        test_ratio=arguments.test_ratio,
    )

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    train_texts = [record.text for record in split_map["train"]]
    idf_dictionary = compute_idf_dictionary(
        train_texts,
        tokenizer,
        max_length=MAX_VECTOR_LENGTH,
    )

    summary: dict[str, object] = {
        "source": arguments.source,
        "pooling_mode": "tfidf_weightedmean",
        "idf_level": "token_id",
        "model_name": EMBEDDING_MODEL,
        "seed": arguments.seed,
        "splits": {name: summarize_records(split_records_list) for name, split_records_list in split_map.items()},
    }

    for split_name, split_records_list in split_map.items():
        save_records(output_dir / f"{split_name}_records.jsonl", split_records_list)

        pair_seed = arguments.seed + len(split_name)
        pairs = build_pairs(
            split_records_list,
            seed=pair_seed,
            max_positive_pairs_per_class=arguments.max_positive_pairs_per_class,
            negative_pairs_per_positive=arguments.negative_pairs_per_positive,
        )
        save_jsonl(output_dir / f"{split_name}_pairs.jsonl", pairs)

        triplets = build_triplets(
            split_records_list,
            seed=pair_seed,
            negatives_per_anchor=arguments.negatives_per_anchor,
        )
        save_jsonl(output_dir / f"{split_name}_triplets.jsonl", triplets)

        summary["splits"][split_name] = {
            **summary["splits"][split_name],
            "pairs": len(pairs),
            "triplets": len(triplets),
        }

    save_idf_dictionary(
        output_dir / "idf_token_id.json",
        idf_dictionary=idf_dictionary,
        model_name=EMBEDDING_MODEL,
        document_count=len(train_texts),
    )
    summary["idf_tokens"] = len(idf_dictionary)

    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
