"""Evaluate baseline and TF-IDF pooling variants for multilingual-e5-large-instruct."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from e5 import E5Model
from e5_experiment_data import TicketRecord, load_records


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline mean pooling, TF-IDF pooling without finetuning, and TF-IDF pooling after finetuning."
    )
    parser.add_argument("--dataset-dir", default="data/e5_pooling")
    parser.add_argument("--checkpoint-path", default="data/e5_pooling/checkpoints/best_pooling_checkpoint.pt")
    parser.add_argument("--idf-path", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha-init", type=float, default=1.0)
    parser.add_argument("--output-path", default="data/e5_pooling/evaluation_comparison.json")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    return parser.parse_args()


def retrieval_metrics(
    candidate_embeddings: torch.Tensor,
    candidate_records: Sequence[TicketRecord],
    query_embeddings: torch.Tensor,
    query_records: Sequence[TicketRecord],
    *,
    top_k: int = 3,
) -> dict[str, float]:
    if not candidate_records or not query_records:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "mrr": 0.0}

    similarity_matrix = query_embeddings @ candidate_embeddings.T
    candidate_labels = [record.class_id for record in candidate_records]

    top1_hits = 0
    topk_hits = 0
    reciprocal_rank_sum = 0.0

    for query_index, query_record in enumerate(query_records):
        ranking = torch.argsort(similarity_matrix[query_index], descending=True)
        ranked_labels = [candidate_labels[index] for index in ranking.tolist()]
        if ranked_labels[0] == query_record.class_id:
            top1_hits += 1
        if query_record.class_id in ranked_labels[:top_k]:
            topk_hits += 1
        for rank, label in enumerate(ranked_labels, start=1):
            if label == query_record.class_id:
                reciprocal_rank_sum += 1.0 / rank
                break

    query_count = len(query_records)
    return {
        "top1_accuracy": top1_hits / query_count,
        "top3_accuracy": topk_hits / query_count,
        "mrr": reciprocal_rank_sum / query_count,
    }


def evaluate_mode(
    *,
    dataset_dir: Path,
    candidate_records: Sequence[TicketRecord],
    query_split_names: Sequence[str],
    pooling_mode: str,
    idf_path: str,
    checkpoint_path: str | None,
    alpha_init: float,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    if checkpoint_path and not Path(checkpoint_path).exists():
        return {
            split_name: {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "mrr": 0.0,
            }
            for split_name in query_split_names
        }

    embedder = E5Model()
    embedder.load(
        pooling_mode=pooling_mode,
        idf_path=idf_path if pooling_mode == "tfidf_weightedmean" else "",
        checkpoint_path=checkpoint_path,
        alpha_init=alpha_init,
        quantize=False,
    )

    candidate_embeddings = embedder.encode_texts(
        [record.text for record in candidate_records],
        batch_size=batch_size,
    )

    results: dict[str, dict[str, float]] = {}
    for split_name in query_split_names:
        query_records = load_records(dataset_dir / f"{split_name}_records.jsonl")
        query_embeddings = embedder.encode_texts(
            [record.text for record in query_records],
            batch_size=batch_size,
        )
        results[split_name] = retrieval_metrics(
            candidate_embeddings,
            candidate_records,
            query_embeddings,
            query_records,
        )
    return results


def main() -> None:
    arguments = parse_arguments()
    dataset_dir = Path(arguments.dataset_dir)
    candidate_records = load_records(dataset_dir / "train_records.jsonl")
    idf_path = arguments.idf_path or str(dataset_dir / "idf_token_id.json")

    comparison = {
        "candidate_split": "train",
        "query_splits": list(arguments.splits),
        "modes": {
            "baseline_mean_pooling": evaluate_mode(
                dataset_dir=dataset_dir,
                candidate_records=candidate_records,
                query_split_names=arguments.splits,
                pooling_mode="mean",
                idf_path="",
                checkpoint_path=None,
                alpha_init=arguments.alpha_init,
                batch_size=arguments.batch_size,
            ),
            "tfidf_pooling_zero_shot": evaluate_mode(
                dataset_dir=dataset_dir,
                candidate_records=candidate_records,
                query_split_names=arguments.splits,
                pooling_mode="tfidf_weightedmean",
                idf_path=idf_path,
                checkpoint_path=None,
                alpha_init=arguments.alpha_init,
                batch_size=arguments.batch_size,
            ),
            "tfidf_pooling_finetuned": evaluate_mode(
                dataset_dir=dataset_dir,
                candidate_records=candidate_records,
                query_split_names=arguments.splits,
                pooling_mode="tfidf_weightedmean",
                idf_path=idf_path,
                checkpoint_path=arguments.checkpoint_path,
                alpha_init=arguments.alpha_init,
                batch_size=arguments.batch_size,
            ),
        },
    }

    output_path = Path(arguments.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
