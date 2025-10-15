import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Инициализация FastAPI
app = FastAPI()

# Загрузка модели
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')

# Функция усреднения эмбеддингов
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Получение вектора для запроса
def get_query_vector(query):
    query_vector = tokenizer(query, return_tensors='pt', max_length=512, truncation=True, padding=True)

    with torch.no_grad():
        query_output = model(**query_vector)
        query_embedding = average_pool(query_output.last_hidden_state, query_vector['attention_mask'])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

    return query_embedding

# Модель для Pydantic (для валидации входных данных)
class QueryRequest(BaseModel):
    query: str



@app.get("/health")
async def health_check():
    return {"status": "ok", "model": f"{model}"}

@app.post("/generate_vector" )
async def generate_vector(query_data: QueryRequest,is_docker:bool=True):
    query = query_data.query
    

    if not query:
        raise HTTPException(status_code=400, detail="Query text is required")

    # Получение эмбеддинга для запроса
    query_embedding = get_query_vector(query)

    # Отправка запроса на сервер FAISS для поиска
    # Сервис vector_db работает в контейнере с именем «vector_db»
    if is_docker:
        faiss_server_url = "http://vector_db:5004/ticket/search"
    else:
        faiss_server_url = "http://localhost:5004/ticket/search"
    headers = {"Content-Type": "application/json"}
    # print(query_embedding.tolist())

    response = requests.post(faiss_server_url, json={ "query_vector": query_embedding.tolist()[0 ]}, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=500, detail="Error from FAISS server")



@app.post("/get_vector")
async def generate_vector(query_data: QueryRequest):
    query = query_data.query
    query_embedding = get_query_vector(query)
    vector_list = query_embedding.squeeze().tolist()
    return {"vector" : vector_list}


# Для запуска через uvicorn:
# uvicorn e5:app --host 0.0.0.0 --port 5003 --reload