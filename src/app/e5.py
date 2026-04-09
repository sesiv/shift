"""Embedding utilities for multilingual-e5-large-instruct."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoConfig, AutoTokenizer

from consts import (
    EMBEDDING_IDF_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_POOLING_ALPHA,
    EMBEDDING_POOLING_CHECKPOINT,
    EMBEDDING_POOLING_MODE,
    EMBEDDING_QUANTIZE,
    MAX_VECTOR_LENGTH,
)
from modeling_xlm_roberta import TfidfWeightedMeanPooling, XLMRobertaE5Model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_idf_artifact(path: str | Path | None) -> dict[int, float]:
    if not path:
        return {}

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "idf" in payload:
        payload = payload["idf"]
    return {int(token_id): float(weight) for token_id, weight in payload.items()}


def load_pooling_checkpoint(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def create_sentence_encoder(
    *,
    model_name: str = EMBEDDING_MODEL,
    pooling_mode: str = "mean",
    idf_path: str | Path | None = None,
    alpha_init: float = 1.0,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> tuple[Any, XLMRobertaE5Model]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    idf_weights = load_idf_artifact(idf_path)

    if pooling_mode == "tfidf_weightedmean" and not idf_weights:
        raise ValueError(
            "TF-IDF pooling requires a non-empty IDF artifact built from the train corpus."
        )

    model = XLMRobertaE5Model.from_pretrained(
        model_name,
        config=config,
        pooling_mode=pooling_mode,
        alpha_init=alpha_init,
        idf_weights=idf_weights,
    )

    checkpoint_payload = load_pooling_checkpoint(checkpoint_path)
    if checkpoint_payload:
        apply_checkpoint_to_sentence_encoder(model, checkpoint_payload)

    model.to(device)
    model.eval()
    return tokenizer, model


def apply_checkpoint_to_sentence_encoder(
    model: XLMRobertaE5Model,
    checkpoint_payload: dict[str, Any],
) -> None:
    pooling_state = checkpoint_payload.get("pooling_state_dict")
    if pooling_state:
        model.pooling.load_state_dict(pooling_state, strict=False)

    last_block_state = checkpoint_payload.get("last_transformer_block_state_dict")
    if last_block_state:
        model.get_last_transformer_block().load_state_dict(last_block_state)


class E5Model:
    """Runtime wrapper for generating E5 vectors."""

    def __init__(self) -> None:
        self.tokenizer = None
        self.model: XLMRobertaE5Model | None = None
        self.device = "cpu"

    def load(
        self,
        *,
        pooling_mode: str | None = None,
        idf_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        alpha_init: float | None = None,
        quantize: bool | None = None,
        device: str | None = None,
    ) -> None:
        """Load tokenizer and embedding model."""

        selected_pooling_mode = pooling_mode or EMBEDDING_POOLING_MODE
        selected_idf_path = EMBEDDING_IDF_PATH if idf_path is None else idf_path
        selected_checkpoint_path = (
            EMBEDDING_POOLING_CHECKPOINT if checkpoint_path is None else checkpoint_path
        )
        selected_alpha = EMBEDDING_POOLING_ALPHA if alpha_init is None else alpha_init
        selected_quantize = EMBEDDING_QUANTIZE if quantize is None else quantize
        self.device = device or "cpu"

        logger.info(
            "Loading local XLM-Roberta E5 model with pooling=%s checkpoint=%s",
            selected_pooling_mode,
            selected_checkpoint_path or "<none>",
        )

        self.tokenizer, self.model = create_sentence_encoder(
            model_name=EMBEDDING_MODEL,
            pooling_mode=selected_pooling_mode,
            idf_path=selected_idf_path,
            alpha_init=selected_alpha,
            checkpoint_path=selected_checkpoint_path,
            device=self.device,
        )

        if selected_quantize and self.device == "cpu":
            self.model.encoder = torch.quantization.quantize_dynamic(
                self.model.encoder,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

        self.model.eval()
        logger.info("Local XLM-Roberta E5 model loaded")

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
    ) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for start_index in range(0, len(texts), batch_size):
                batch_texts = list(texts[start_index : start_index + batch_size])
                encoded_batch = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=MAX_VECTOR_LENGTH,
                    truncation=True,
                    padding=True,
                )
                encoded_batch = {
                    name: tensor.to(self.device) for name, tensor in encoded_batch.items()
                }
                outputs = self.model(**encoded_batch)
                embeddings.append(outputs.sentence_embedding.cpu())

        if not embeddings:
            return torch.empty((0, int(self.model.config.hidden_size)), dtype=torch.float32)
        return torch.cat(embeddings, dim=0)

    def generate_vector(self, query: str) -> list[float]:
        embedding = self.encode_texts([query], batch_size=1)
        return embedding.squeeze(0).tolist()
