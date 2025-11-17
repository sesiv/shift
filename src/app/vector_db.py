from typing import List, Dict, Any
import chromadb
from fastapi import FastAPI, HTTPException
import logging
import json
from chromadb.config import Settings
import os
from utils import distance_to_confidence
from e5 import E5Model
from consts import (
    SCORE_DELTA,
    N_RESULTS,
    CHROMA_CLIENT_AUTH_PROVIDER,
    CHROMA_SERVER_PORT,
    CHROMA_SERVER_HOST,
    CHROMA_COLLECTION_NAME,
 )
from schemas import TicketPayload

logging.basicConfig(level=logging.INFO)



# загружаем зависимости при старте
e5_instance = E5Model()
collection = None

try:
    logging.info("[vector_db] загрузка e5")
    e5_instance.load()
    logging.info("[vector_db] e5 загружена")

    # подключение к chromadb
    client = chromadb.HttpClient(
        host=CHROMA_SERVER_HOST,
        port=CHROMA_SERVER_PORT,
        settings=Settings(
            chroma_client_auth_provider=CHROMA_CLIENT_AUTH_PROVIDER,
            chroma_client_auth_credentials=os.environ["CHROMA_CLIENT_AUTH_CREDENTIALS"],
        ),
    )
    collection = client.get_collection(CHROMA_COLLECTION_NAME)

    logging.info("[vector_db] сервис готов")
except Exception as e:
    logging.exception("[vector_db] ошибка инициализации: %s", e)

# Инициализация FastAPI
app = FastAPI()


async def search_ticket(message, n_results):

    query_vector = e5_instance.generate_vector(message)

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

    embedding = e5_instance.generate_vector(document)

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


@app.post("/aggregate", status_code=200)
async def aggregate_nodes(state: str, message: str) -> dict:
    """
    Классифицирует сообщение в один из узлов и вычисляет откалиброванную уверенность.

    Args:
        state: Текущее состояние разговора
        message: Сообщение пользователя для классификации

    Returns:
        Словарь с ключами:
        - predicted_id: str | None — предсказанный идентификатор узла
        - confidence: float в диапазоне [0,1] — уверенность в предсказании
        - top_categories: список словарей {id, score}, отсортированных по убыванию score
        - best_distance: float | None — минимальная дистанция для предсказанной категории
    """
    similar_nodes = await search_ticket(message=message, n_results=N_RESULTS)
    similar_nodes_dict = json.loads(similar_nodes.get("response", "[]"))
    logging.info(f"similar_nodes_dict {similar_nodes_dict}")

    hits: dict = {"folder": {}, "slmService": {}, "categoriesWork": {}}

    if state == "baseState":
        for node in similar_nodes_dict:
            hits["folder"][node["folder"]] = (
                hits["folder"].get(node["folder"], 0) + node["distance"]
            )
            hits["slmService"][node["slmService"]] = (
                hits["slmService"].get(node["slmService"], 0) + node["distance"]
            )
            hits["categoriesWork"][node["categoriesWork"]] = (
                hits["categoriesWork"].get(node["categoriesWork"], 0) + node["distance"]
            )
    elif state == "folder":
        for node in similar_nodes_dict:
            hits["slmService"][node["slmService"]] = (
                hits["slmService"].get(node["slmService"], 0) + node["distance"]
            )
            hits["categoriesWork"][node["categoriesWork"]] = (
                hits["categoriesWork"].get(node["categoriesWork"], 0) + node["distance"]
            )
    elif state == "slmService":
        for node in similar_nodes_dict:
            hits["categoriesWork"][node["categoriesWork"]] = (
                hits["categoriesWork"].get(node["categoriesWork"], 0) + node["distance"]
            )

    logging.info(f"hits {hits}")

    best = {
        "folder": {"id": "", "score": 0.0},
        "slmService": {"id": "", "score": 0.0},
        "categoriesWork": {"id": "", "score": 0.0},
    }

    for level in hits:
        for hit in hits[level]:
            if best[level]["score"] < hits[level][hit]:
                best[level]["score"] = hits[level][hit]
                best[level]["id"] = hit

    max_score = max(best[level]["score"] for level in best)
    logging.info(f"max score {max_score}")
    logging.info(f"best {best}")

    predicted_id = None
    # Приоритетно выбираем id с максимальным score среди уровней: категории работ, услуга, папка
    for priority in ["categoriesWork", "slmService", "folder"]:
        if abs(best[priority]["score"] - max_score) < SCORE_DELTA:
            predicted_id = best[priority]["id"]
            break

    # Определяем минимальную дистанцию для предсказанной категории для калибровки уверенности
    best_distance = None
    if predicted_id:
        candidate_distances: List[float] = []
        # Собираем дистанции для нод, у которых совпадает predicted_id на любом уровне
        for node in similar_nodes_dict:
            if (
                node.get("categoriesWork") == predicted_id
                or node.get("slmService") == predicted_id
                or node.get("folder") == predicted_id
            ):
                try:
                    candidate_distances.append(float(node["distance"]))
                except Exception:
                    pass
        if candidate_distances:
            best_distance = min(candidate_distances)
        else:
            # Если не найдено — используем минимальную дистанцию среди всех нод
            try:
                best_distance = min(float(n["distance"]) for n in similar_nodes_dict)
            except Exception:
                best_distance = None

    confidence = (
        distance_to_confidence(best_distance) if best_distance is not None else 0.0
    )

    # Формируем топ категорий (по убыванию агрегированного score) для подсказок
    categories_scores = hits["categoriesWork"]
    sorted_categories = sorted(
        categories_scores.items(), key=lambda x: x[1], reverse=True
    )
    top_categories = [{"id": cid, "score": score} for cid, score in sorted_categories]

    result = {
        "predicted_id": predicted_id,
        "confidence": confidence,
        "top_categories": top_categories,
        "best_distance": best_distance,
    }
    return result


@app.get("/health")
async def health_check():
    """
    роверяем, что стартовая инициализация завершилась.
    """
    if collection:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="loading")


# Для запуска через uvicorn:
# uvicorn vector_db:app --host 0.0.0.0 --port 5004 --reload

