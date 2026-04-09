"""Local fork of XLM-Roberta with sentence-embedding pooling for E5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel


@dataclass
class XLMRobertaE5Output(ModelOutput):
    sentence_embedding: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class TfidfWeightedMeanPooling(nn.Module):
    """Mean pooling with optional TF-IDF token weights."""

    def __init__(
        self,
        *,
        mode: str = "mean",
        alpha_init: float = 1.0,
        vocab_size: int = 1,
        idf_weights: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.register_buffer(
            "idf_lookup",
            torch.zeros(max(int(vocab_size), 1), dtype=torch.float32),
            persistent=True,
        )

        if idf_weights:
            self.set_idf_weights(idf_weights, vocab_size=vocab_size)

    def set_idf_weights(
        self,
        idf_weights: dict[int, float],
        *,
        vocab_size: int | None = None,
    ) -> None:
        required_vocab_size = max(idf_weights) + 1 if idf_weights else 1
        target_vocab_size = max(vocab_size or 1, required_vocab_size)
        lookup = torch.zeros(target_vocab_size, dtype=torch.float32)
        for token_id, weight in idf_weights.items():
            if token_id < 0 or token_id >= target_vocab_size:
                continue
            lookup[token_id] = float(weight)
        self.idf_lookup = lookup

    def _token_weights(
        self,
        *,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        attention_mask_float = attention_mask.to(dtype=dtype)
        if self.mode == "mean":
            return attention_mask_float

        if self.mode != "tfidf_weightedmean":
            raise ValueError(f"Unsupported pooling mode: {self.mode}")
        if input_ids is None:
            raise ValueError("input_ids are required for tfidf_weightedmean pooling")

        lookup = self.idf_lookup
        if lookup.numel() == 0:
            raise ValueError("IDF weights are not initialized for tfidf_weightedmean pooling")
        if input_ids.max().item() >= lookup.shape[0]:
            raise ValueError(
                "Encountered token id outside the loaded IDF dictionary. "
                "Rebuild the IDF artifact with the current tokenizer."
            )

        lookup = lookup.to(device=input_ids.device)
        weights = torch.zeros_like(attention_mask_float)
        alpha = self.alpha.to(dtype=dtype)

        for row_index in range(input_ids.shape[0]):
            valid_mask = attention_mask[row_index].bool()
            token_ids = input_ids[row_index, valid_mask]
            if token_ids.numel() == 0:
                continue

            _, inverse_indices, counts = torch.unique(
                token_ids,
                sorted=False,
                return_inverse=True,
                return_counts=True,
            )
            token_frequency = counts[inverse_indices].to(dtype=dtype) / float(token_ids.numel())
            inverse_document_frequency = lookup[token_ids].to(dtype=dtype)
            tfidf_weights = token_frequency * inverse_document_frequency
            weights[row_index, valid_mask] = 1.0 + alpha * tfidf_weights

        return weights * attention_mask_float

    def forward(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = self._token_weights(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=last_hidden_states.dtype,
        )
        weighted_hidden_states = last_hidden_states * weights.unsqueeze(-1)
        normalization = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return weighted_hidden_states.sum(dim=1) / normalization


class XLMRobertaE5Model(XLMRobertaModel):
    """Local fork of XLMRobertaModel that produces sentence embeddings."""

    _keys_to_ignore_on_load_missing = [r"pooling\.alpha", r"pooling\.idf_lookup"]
    _keys_to_ignore_on_load_unexpected = [r"pooler\."]

    def __init__(
        self,
        config: Any,
        *,
        pooling_mode: str = "mean",
        alpha_init: float = 1.0,
        idf_weights: dict[int, float] | None = None,
    ) -> None:
        super().__init__(config, add_pooling_layer=False)
        self.pooling = TfidfWeightedMeanPooling(
            mode=pooling_mode,
            alpha_init=alpha_init,
            vocab_size=int(getattr(config, "vocab_size", 0) or 1),
            idf_weights=idf_weights,
        )

    def set_pooling_mode(self, mode: str) -> None:
        self.pooling.mode = mode

    def set_idf_weights(self, idf_weights: dict[int, float]) -> None:
        self.pooling.set_idf_weights(idf_weights, vocab_size=int(self.config.vocab_size))

    def freeze_encoder(self) -> None:
        for name, parameter in self.named_parameters():
            if not name.startswith("pooling."):
                parameter.requires_grad = False

    def get_last_transformer_block(self) -> nn.Module:
        return self.encoder.layer[-1]

    def unfreeze_last_transformer_block(self) -> nn.Module:
        last_block = self.get_last_transformer_block()
        for parameter in last_block.parameters():
            parameter.requires_grad = True
        return last_block

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> XLMRobertaE5Output | tuple[torch.Tensor, ...]:
        base_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )

        if attention_mask is None:
            if input_ids is not None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = torch.ones(
                    base_outputs.last_hidden_state.shape[:2],
                    device=base_outputs.last_hidden_state.device,
                    dtype=torch.long,
                )

        sentence_embedding = self.pooling(
            base_outputs.last_hidden_state,
            attention_mask,
            input_ids,
        )
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)

        if return_dict is False:
            return (
                sentence_embedding,
                base_outputs.last_hidden_state,
                base_outputs.hidden_states,
                base_outputs.attentions,
            )

        return XLMRobertaE5Output(
            sentence_embedding=sentence_embedding,
            last_hidden_state=base_outputs.last_hidden_state,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )
