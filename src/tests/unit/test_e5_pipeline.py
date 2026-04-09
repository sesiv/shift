import math
from pathlib import Path
import sys
import unittest

import torch
from transformers import XLMRobertaConfig


APP_DIR = Path(__file__).resolve().parents[2] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from e5_experiment_data import (  # noqa: E402
    TicketRecord,
    build_pairs,
    build_triplets,
    compute_idf_dictionary,
    split_records,
)
from modeling_xlm_roberta import TfidfWeightedMeanPooling, XLMRobertaE5Model  # noqa: E402
from xlsx_reader import read_xlsx_records  # noqa: E402


class FakeTokenizer:
    all_special_ids = [0]

    def __call__(self, text, **kwargs):
        mapping = {
            "alpha beta": [101, 11, 12, 0],
            "alpha alpha": [101, 11, 11, 0],
        }
        return {"input_ids": mapping[text]}


def make_record(record_id: str, class_id: str, text: str) -> TicketRecord:
    return TicketRecord(
        record_id=record_id,
        class_id=class_id,
        class_name=class_id,
        service_id="service",
        service_name="service",
        request_type_id="request",
        request_type_name="request",
        subject=text,
        description=text,
        text=text,
    )


class PoolingTests(unittest.TestCase):
    def test_tfidf_pooling_respects_attention_mask_and_weights(self):
        pooling = TfidfWeightedMeanPooling(
            mode="tfidf_weightedmean",
            alpha_init=1.0,
            vocab_size=4,
            idf_weights={1: 2.0, 2: 1.0},
        )
        hidden_states = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [10.0, 10.0]]])
        input_ids = torch.tensor([[1, 2, 0]])
        attention_mask = torch.tensor([[1, 1, 0]])

        pooled = pooling(hidden_states, attention_mask, input_ids)

        self.assertTrue(
            torch.allclose(
                pooled,
                torch.tensor([[2.0 / 3.5, 1.5 / 3.5]]),
                atol=1e-6,
            )
        )

    def test_local_xlm_roberta_model_returns_sentence_embedding(self):
        config = XLMRobertaConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        )
        model = XLMRobertaE5Model(
            config,
            pooling_mode="tfidf_weightedmean",
            alpha_init=1.0,
            idf_weights={4: 2.0, 5: 1.0, 6: 1.5},
        )

        outputs = model(
            input_ids=torch.tensor([[4, 5, 6, 1]]),
            attention_mask=torch.tensor([[1, 1, 1, 0]]),
        )

        self.assertEqual(outputs.sentence_embedding.shape, (1, 16))
        self.assertEqual(outputs.last_hidden_state.shape, (1, 4, 16))


class DatasetPreparationTests(unittest.TestCase):
    def test_split_keeps_minimum_train_examples(self):
        records = [
            make_record("a1", "A", "text a1"),
            make_record("a2", "A", "text a2"),
            make_record("a3", "A", "text a3"),
            make_record("a4", "A", "text a4"),
            make_record("b1", "B", "text b1"),
            make_record("b2", "B", "text b2"),
            make_record("c1", "C", "text c1"),
        ]

        split_map = split_records(records, seed=7, val_ratio=0.25, test_ratio=0.25)

        train_by_label = {}
        for record in split_map["train"]:
            train_by_label[record.class_id] = train_by_label.get(record.class_id, 0) + 1

        self.assertGreaterEqual(train_by_label["A"], 2)
        self.assertEqual(train_by_label["B"], 2)
        self.assertEqual(train_by_label["C"], 1)

    def test_pairs_and_triplets_use_class_boundaries(self):
        records = [
            make_record("a1", "A", "text a1"),
            make_record("a2", "A", "text a2"),
            make_record("b1", "B", "text b1"),
            make_record("b2", "B", "text b2"),
        ]

        pairs = build_pairs(records, seed=1, max_positive_pairs_per_class=10)
        triplets = build_triplets(records, seed=1)

        positive_pairs = [pair for pair in pairs if pair["pair_type"] == "positive"]
        negative_pairs = [pair for pair in pairs if pair["pair_type"] == "negative"]

        self.assertTrue(positive_pairs)
        self.assertTrue(negative_pairs)
        self.assertTrue(triplets)
        for pair in positive_pairs:
            self.assertEqual(pair["anchor_class_id"], pair["other_class_id"])
        for pair in negative_pairs:
            self.assertNotEqual(pair["anchor_class_id"], pair["other_class_id"])
        for triplet in triplets:
            self.assertEqual(triplet["anchor_class_id"], "A" if triplet["anchor_id"].startswith("a") else "B")
            self.assertNotEqual(triplet["anchor_class_id"], triplet["negative_class_id"])

    def test_idf_is_computed_only_from_unique_tokens_per_document(self):
        tokenizer = FakeTokenizer()
        idf = compute_idf_dictionary(
            ["alpha beta", "alpha alpha"],
            tokenizer,
            max_length=8,
        )

        self.assertTrue(math.isclose(idf[11], 1.0, rel_tol=1e-6))
        self.assertTrue(math.isclose(idf[12], math.log(1.5) + 1.0, rel_tol=1e-6))


class XlsxReaderTests(unittest.TestCase):
    def test_reader_extracts_expected_headers(self):
        records = read_xlsx_records(APP_DIR.parent / "data" / "ExportSDLab.xlsx")

        self.assertTrue(records)
        self.assertIn("UUID Категории работ", records[0])
        self.assertIn("Описание", records[0])


if __name__ == "__main__":
    unittest.main()
