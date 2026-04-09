"""Dataset preparation utilities for TF-IDF E5 pooling experiments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence

from xlsx_reader import read_xlsx_records


WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class TicketRecord:
    record_id: str
    class_id: str
    class_name: str
    service_id: str
    service_name: str
    request_type_id: str
    request_type_name: str
    subject: str
    description: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "TicketRecord":
        return cls(**payload)


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def compose_ticket_text(subject: str | None, description: str | None) -> str:
    normalized_subject = normalize_text(subject)
    normalized_description = normalize_text(description)

    if normalized_subject and normalized_description:
        if normalized_subject.lower() in normalized_description.lower():
            return normalized_description
        return f"{normalized_subject}. {normalized_description}"
    return normalized_subject or normalized_description


def load_ticket_records(xlsx_path: str | Path) -> list[TicketRecord]:
    records = read_xlsx_records(xlsx_path)
    ticket_records: list[TicketRecord] = []

    for row_index, row in enumerate(records, start=2):
        class_id = normalize_text(row.get("UUID Категории работ"))
        class_name = normalize_text(row.get("Имя Категории работ"))
        text = compose_ticket_text(row.get("Тема"), row.get("Описание"))

        if not class_id or not text:
            continue

        ticket_records.append(
            TicketRecord(
                record_id=f"row_{row_index}",
                class_id=class_id,
                class_name=class_name,
                service_id=normalize_text(row.get("UUID Услуги")),
                service_name=normalize_text(row.get("Имя услуги")),
                request_type_id=normalize_text(row.get("ID типа запроса")),
                request_type_name=normalize_text(row.get("Имя запроса")),
                subject=normalize_text(row.get("Тема")),
                description=normalize_text(row.get("Описание")),
                text=text,
            )
        )

    return ticket_records


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        return [json.loads(line) for line in file_handle if line.strip()]


def save_records(path: str | Path, records: Sequence[TicketRecord]) -> None:
    save_jsonl(path, (record.to_dict() for record in records))


def load_records(path: str | Path) -> list[TicketRecord]:
    return [TicketRecord.from_dict(payload) for payload in load_jsonl(path)]


def group_by_class(records: Sequence[TicketRecord]) -> dict[str, list[TicketRecord]]:
    grouped: dict[str, list[TicketRecord]] = defaultdict(list)
    for record in records:
        grouped[record.class_id].append(record)
    return grouped


def split_records(
    records: Sequence[TicketRecord],
    *,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_train_examples_per_class: int = 2,
) -> dict[str, list[TicketRecord]]:
    """Split records per class while keeping enough train examples for positives."""

    rng = random.Random(seed)
    grouped_records = group_by_class(records)
    split_map = {"train": [], "validation": [], "test": []}

    for class_id in sorted(grouped_records):
        class_records = list(grouped_records[class_id])
        rng.shuffle(class_records)
        class_size = len(class_records)

        train_count = class_size
        validation_count = 0
        test_count = 0

        if class_size >= min_train_examples_per_class + 1:
            validation_count = int(round(class_size * val_ratio)) if val_ratio else 0
            test_count = int(round(class_size * test_ratio)) if test_ratio else 0

            if class_size >= min_train_examples_per_class + 3 and validation_count == 0 and val_ratio > 0:
                validation_count = 1
            if class_size >= min_train_examples_per_class + 4 and test_count == 0 and test_ratio > 0:
                test_count = 1

            while class_size - validation_count - test_count < min_train_examples_per_class:
                if test_count >= validation_count and test_count > 0:
                    test_count -= 1
                elif validation_count > 0:
                    validation_count -= 1
                else:
                    break

            train_count = class_size - validation_count - test_count

        split_map["train"].extend(class_records[:train_count])
        split_map["validation"].extend(
            class_records[train_count : train_count + validation_count]
        )
        split_map["test"].extend(class_records[train_count + validation_count :])

    return split_map


def _sample_positive_pairs_for_class(
    records: Sequence[TicketRecord],
    *,
    rng: random.Random,
    max_pairs: int,
) -> list[tuple[TicketRecord, TicketRecord]]:
    if len(records) < 2:
        return []

    total_pairs = len(records) * (len(records) - 1) // 2
    if total_pairs <= max_pairs:
        return list(combinations(records, 2))

    chosen_pairs: set[tuple[int, int]] = set()
    while len(chosen_pairs) < max_pairs:
        left_index, right_index = sorted(rng.sample(range(len(records)), 2))
        chosen_pairs.add((left_index, right_index))

    return [(records[left_index], records[right_index]) for left_index, right_index in sorted(chosen_pairs)]


def build_pairs(
    records: Sequence[TicketRecord],
    *,
    seed: int = 42,
    max_positive_pairs_per_class: int = 200,
    negative_pairs_per_positive: int = 1,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped_records = group_by_class(records)
    labels = sorted(grouped_records)
    pairs: list[dict[str, Any]] = []

    for class_id in labels:
        class_records = grouped_records[class_id]
        positive_pairs = _sample_positive_pairs_for_class(
            class_records,
            rng=rng,
            max_pairs=max_positive_pairs_per_class,
        )

        other_labels = [label for label in labels if label != class_id and grouped_records[label]]
        if not other_labels:
            continue

        for anchor, positive in positive_pairs:
            pairs.append(
                {
                    "pair_type": "positive",
                    "label": 1,
                    "anchor_id": anchor.record_id,
                    "anchor_text": anchor.text,
                    "anchor_class_id": anchor.class_id,
                    "other_id": positive.record_id,
                    "other_text": positive.text,
                    "other_class_id": positive.class_id,
                }
            )

            for _ in range(negative_pairs_per_positive):
                negative_label = rng.choice(other_labels)
                negative = rng.choice(grouped_records[negative_label])
                pairs.append(
                    {
                        "pair_type": "negative",
                        "label": 0,
                        "anchor_id": anchor.record_id,
                        "anchor_text": anchor.text,
                        "anchor_class_id": anchor.class_id,
                        "other_id": negative.record_id,
                        "other_text": negative.text,
                        "other_class_id": negative.class_id,
                    }
                )

    return pairs


def build_triplets(
    records: Sequence[TicketRecord],
    *,
    seed: int = 42,
    negatives_per_anchor: int = 1,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped_records = group_by_class(records)
    labels = [label for label, group in grouped_records.items() if len(group) >= 2]
    triplets: list[dict[str, Any]] = []

    for class_id in labels:
        class_records = grouped_records[class_id]
        negative_labels = [label for label in grouped_records if label != class_id and grouped_records[label]]
        if not negative_labels:
            continue

        for anchor in class_records:
            positive_candidates = [record for record in class_records if record.record_id != anchor.record_id]
            for _ in range(negatives_per_anchor):
                positive = rng.choice(positive_candidates)
                negative_label = rng.choice(negative_labels)
                negative = rng.choice(grouped_records[negative_label])
                triplets.append(
                    {
                        "anchor_id": anchor.record_id,
                        "anchor_text": anchor.text,
                        "anchor_class_id": anchor.class_id,
                        "positive_id": positive.record_id,
                        "positive_text": positive.text,
                        "negative_id": negative.record_id,
                        "negative_text": negative.text,
                        "negative_class_id": negative.class_id,
                    }
                )

    return triplets


def compute_idf_dictionary(
    texts: Sequence[str],
    tokenizer: Any,
    *,
    max_length: int,
) -> dict[int, float]:
    document_frequency: Counter[int] = Counter()
    special_token_ids = set(getattr(tokenizer, "all_special_ids", []))

    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
        unique_token_ids = {
            int(token_id)
            for token_id in encoded["input_ids"]
            if int(token_id) not in special_token_ids
        }
        document_frequency.update(unique_token_ids)

    total_documents = max(len(texts), 1)
    return {
        token_id: math.log((1 + total_documents) / (1 + frequency)) + 1.0
        for token_id, frequency in document_frequency.items()
    }


def save_idf_dictionary(
    path: str | Path,
    *,
    idf_dictionary: dict[int, float],
    model_name: str,
    document_count: int,
) -> None:
    serialized = {
        "token_level": "token_id",
        "model_name": model_name,
        "document_count": document_count,
        "idf": {str(token_id): weight for token_id, weight in idf_dictionary.items()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")


def load_idf_dictionary(path: str | Path) -> dict[int, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "idf" in payload:
        payload = payload["idf"]
    return {int(token_id): float(weight) for token_id, weight in payload.items()}


def summarize_records(records: Sequence[TicketRecord]) -> dict[str, Any]:
    grouped_records = group_by_class(records)
    class_sizes = [len(group) for group in grouped_records.values()]

    return {
        "records": len(records),
        "classes": len(grouped_records),
        "classes_with_at_least_2_examples": sum(size >= 2 for size in class_sizes),
        "classes_with_at_least_3_examples": sum(size >= 3 for size in class_sizes),
        "min_class_size": min(class_sizes) if class_sizes else 0,
        "max_class_size": max(class_sizes) if class_sizes else 0,
    }
