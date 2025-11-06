import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

# Инициализация FastAPI
app = FastAPI()

# Загрузка модели
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL)

# Функция усреднения эмбеддингов
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Получение вектора для запроса
def generate_vector(query: str) -> list[float]:
    query_vector = tokenizer(query, return_tensors='pt', max_length=512, truncation=True, padding=True)

    with torch.no_grad():
        query_output = model(**query_vector)
        query_embedding = average_pool(query_output.last_hidden_state, query_vector['attention_mask'])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)


    return query_embedding.squeeze().tolist()



@app.get("/health")
async def health_check():
    return {"status": "ok", "model": f"{model}"}


# Для запуска через uvicorn:
# uvicorn e5:app --host 0.0.0.0 --port 5003 --reload