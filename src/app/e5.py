"""
Модуль содержит класс-обёртку для e5
"""

import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from consts import EMBEDDING_MODEL
import logging
from consts import MAX_VECTOR_LENGTH

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class E5Model:
    """
    Обёртка для векторизующей модели e5
    """
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = "cpu"

    def load(self):
        """Загружает модель в память"""
        logging.info("Загрузка модели E5...")
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL)

        self.model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        logging.info("E5 загружена")

    @staticmethod
    def average_pool(last_hidden_states, attention_mask):
        """Усредняет эмбеддинги"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def generate_vector(self, query: str) -> list[float]:
        """Генерирует вектор для запроса"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Токенизация
        query_vector = self.tokenizer(
            query,
            return_tensors='pt',
            max_length=MAX_VECTOR_LENGTH,
            truncation=True,
            padding=True
        )

        # Генерация
        with torch.no_grad():
            query_output = self.model(**query_vector)
            query_embedding = self.average_pool(
                query_output.last_hidden_state,
                query_vector['attention_mask']
            )
            query_embedding = F.normalize(query_embedding, p=2, dim=1)

        return query_embedding.squeeze().tolist()

