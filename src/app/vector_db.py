from typing import List, Dict, Any
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import requests
import logging
import json
from chromadb.config import Settings
import os

from e5 import generate_vector

logging.basicConfig(level=logging.INFO)

client = chromadb.HttpClient(
    host="158.160.17.124",  # IP или домен сервера
    port=8000,              # порт сервера (уточни свой)
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
        chroma_client_auth_credentials=os.environ["CHROMA_CLIENT_AUTH_CREDENTIALS"]
    )
)

collection=client.get_collection("ticketsTrain09082025")

# Инициализация FastAPI
app = FastAPI()


# Модель запроса через Pydantic
class SearchRequest(BaseModel):
    """Модель для запроса поиска тикетов."""
    message: str = Field(..., example="Прошу предоставить права локального администратора...")
    n_results: int = Field(5, gt=0, le=20, description="Количество возвращаемых результатов")


class TicketPayload(BaseModel):
    """
    Модель для данных, необходимых для создания нового тикета.
    """
    categoriesWork: str = Field(..., example="categoriesWork$49302778")
    folder: str = Field(..., example="folder$1115806")
    description: str = Field(..., example="Прошу предоставить права локального администратора...")
    slmService: str = Field(..., example="slmService$1116020")
    name: str = Field(..., example="Предоставление прав локального администратора на ПК")

    model_config = {
        "json_schema_extra": {
            "example": {
                'categoriesWork': 'categoriesWork$49302778',
                'folder': 'folder$1115806',
                'description': 'Прошу предоставить права локального администратора для установки ПО.',
                'slmService': 'slmService$1116020',
                'name': 'Предоставление прав локального администратора на ПК',
            }
        }
    }


@app.post("/ticket/search")
async def search_ticket(query_data: SearchRequest):
    n_results = query_data.n_results
    message = query_data.message

    query_vector = generate_vector(message)

    results=collection.query(
        query_embeddings=query_vector,
        n_results=n_results,
        include=["metadatas","distances","documents"],
    )

    data=[dict() for _ in range(5)]
    # разбираем пути response
    for i,metadata in enumerate(results["metadatas"][0]):
        data[i]['categoriesWork']=metadata["categoriesWork"]
        data[i]['slmService']=metadata["slmService"]
        data[i]['folder']=metadata["folder"]

    for i ,distance in enumerate(results["distances"][0]):
        data[i]["distance"]=distance

    return {"response":json.dumps(data)}


@app.post("/ticket", status_code=201)
async def create_ticket(ticket_data: TicketPayload) -> Dict[str, Any]:
    """
    Создание нового тикета.
    """
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB service is unavailable.")

    new_id = str(collection.count() + 1)
    logging.info(f"Generating new ticket with ID: {new_id}")

    metadata = ticket_data.model_dump()
    metadata['id'] = new_id  # добавляем сгенерированный ID в метаданные

    document = ticket_data.description

    embedding = generate_vector(document)

    #добавляем данные в коллекцию
    try:
        collection.add(
            ids=[new_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata]
        )
        logging.info(f"Successfully added ticket with ID: {new_id}")
        return {"message": "Ticket created successfully", "ticket_id": new_id, "data": metadata}
    except Exception as e:
        logging.error(f"Failed to add ticket to collection: {e}")
        raise HTTPException(status_code=500, detail="Failed to create the ticket in the database.")


@app.delete("/ticket/{ticket_id}")
async def delete_ticket(ticket_id: str) -> Dict[str, str]:
    """
    Удаление тикета по его ID.
    """
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB service is unavailable.")

    logging.info(f"Attempting to delete ticket with ID: {ticket_id}")

    # Проверяем, существует ли такой тикет
    existing_ticket = collection.get(ids=[ticket_id])
    if not existing_ticket or not existing_ticket['ids']:
        logging.warning(f"Ticket with ID {ticket_id} not found.")
        raise HTTPException(status_code=404, detail=f"Ticket with ID '{ticket_id}' not found.")

    # Если тикет существует, удаляем его
    try:
        collection.delete(ids=[ticket_id])
        logging.info(f"Successfully deleted ticket with ID: {ticket_id}")
        return {"message": "Ticket deleted successfully", "ticket_id": ticket_id}
    except Exception as e:
        logging.error(f"Error during ticket deletion: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while deleting the ticket.")




# Для запуска через uvicorn:
# uvicorn vector_db:app --host 0.0.0.0 --port 5004 --reload


