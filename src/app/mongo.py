"""
FastAPI-сервер для работы с документами в MongoDB.

Предоставляет эндпоинты для получения, создания и удаления документов,
хранящихся в коллекции 'nodes' базы данных 'nodes'.
"""
import logging
import os
from typing import List ,Annotated, Optional

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel ,Field
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI")
logging.info(f"Connecting to MongoDB at {MONGO_URI}")

try:
    client = MongoClient(MONGO_URI)
    client.server_info()
    logging.info("Successfully connected to MongoDB.")
except Exception as e:
    logging.error(f"FATAL: Failed to connect to MongoDB: {e}")
    raise SystemExit("MongoDB connection failed") from e

db = client.get_database("nodes")
collection = db.get_collection("nodes")

app = FastAPI(
    title="Document API",
    description="API для управления документами в MongoDB",
    version="1.0.0",
)


class Node(BaseModel):
    """
    Модель документа (ноды), хранящегося в MongoDB.

    Атрибуты:
        id (str): Уникальный идентификатор документа.
        sd_level (str): Уровень категории (например, 'categoriesWork').
        description (str): Подробное описание категории.
        guide (str): Инструкция или руководство по работе с категорией.
        children (list): Список дочерних элементов (обычно пустой).
        path (str): Путь в иерархии (технический).
        name_path (str): Человекочитаемый путь.
        name (str): Название категории.
    """

    id: str = Field(..., example="categoriesWork$48859506")
    sd_level: str = Field(..., example="categoriesWork")
    description: str = Field(
        ...,
        example=(
            'Категория работ "Открытие/изменение овердрафта" включает в себя обращения пользователей, '
            'связанные с техническими и организационными вопросами при работе с сервис-деском...'
        ),
    )
    guide: str = Field(
        ...,
        example=(
            "### Руководство: Открытие или изменение овердрафта\n\n"
            "1. **Заполните заявление.** Скачать форму на корпоративном портале.\n"
            "2. **Укажите детали.** В заявлении укажите лимит и срок.\n"
            "3. **Отправьте на согласование.** Создайте заявку через бота и приложите заявление.\n"
            "4. **Ожидайте уведомления.** Стандартный срок рассмотрения — 3 рабочих дня."
        ),
    )
    children: List = Field(default_factory=list, example=[])
    path: str = Field(
        ...,
        example="/folder$1115801/slmService$1116002/categoriesWork$48859506/"
    )
    name_path: str = Field(
        ...,
        example=(
            "/02. Социальная программа, кадровые вопросы, оплата труда, справки/"
            "Финансы (Справки о доходах, отпускные, расчетный лист, овердрафт)/"
            "Открытие/изменение овердрафта/"
        ),
    )
    name: str = Field(..., example="Открытие/изменение овердрафта")


def decode_id(encoded_id: str) -> str:
    """
    Декодирует идентификатор, заменяя специальную последовательность '#24' на '$'.

    Используется для обхода проблем с URL-кодированием символа '$'.

    Args:
        encoded_id (str): Закодированный идентификатор.

    Returns:
        str: Декодированный идентификатор.
    """
    return encoded_id.replace(r"#24", "$")


@app.get("/document/{id}", response_model=dict)
async def get_document(
    id: str,
    filter_fields: Annotated[
        Optional[str],
        Query(
            description=(
                "Список полей, которые нужно вернуть, через запятую. "
                "Если не указан — возвращается весь объект."
            )
        ),
    ] = None,
) -> JSONResponse:
    """
    Получает документ по его идентификатору.

    Args:
        id (str): Идентификатор документа (может содержать '#24' вместо '$').
        filter_fields (Optional[str]): Список полей для возврата через запятую.

    Returns:
        JSONResponse: Объект документа с указанными полями или полный объект.

    Raises:
        HTTPException: 404 если документ не найден.
    """
    decoded_id = decode_id(id)
    logger.info(f"Fetching document with id: {decoded_id}")

    try:
        node_from_mongo = collection.find_one({"id": decoded_id})
        if not node_from_mongo:
            logger.warning(f"Document with id '{decoded_id}' not found")
            raise HTTPException(status_code=404, detail="Document not found")

        # Удаляем служебное поле MongoDB
        node_from_mongo.pop("_id", None)

        node = Node(**node_from_mongo)

        if filter_fields:
            include_set = set(field.strip() for field in filter_fields.split(","))
            filtered_data = node.model_dump(include=include_set)
            return JSONResponse({"status": 200, "data": filtered_data})

        return JSONResponse({"status": 200, "data": node.model_dump()})

    except PyMongoError as e:
        logger.error(f"Database error while fetching document: {e}")
        raise HTTPException(status_code=500, detail="Database error") from e
    except Exception as e:
        logger.error(f"Unexpected error while fetching document: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/document/{id}", status_code=201)
async def create_document(id: str, node: Node) -> dict:
    """
    Создаёт новый документ с указанным идентификатором.

    Args:
        id (str): Идентификатор документа (может содержать '#24' вместо '$').
        node (Node): Данные нового документа.

    Returns:
        dict: Сообщение об успешном создании и идентификатор.

    Raises:
        HTTPException: 409 если документ с таким id уже существует.
        HTTPException: 500 при ошибке базы данных.
    """
    decoded_id = decode_id(id)
    logger.info(f"Creating document with id: {decoded_id}")

    if collection.find_one({"id": decoded_id}):
        logger.warning(f"Document with id '{decoded_id}' already exists")
        raise HTTPException(status_code=409, detail="Document with this id already exists")

    try:
        node_dict = node.model_dump()
        node_dict["id"] = decoded_id

        result = collection.insert_one(node_dict)
        logger.info(f"Document inserted with MongoDB _id: {result.inserted_id}")
        return {"message": "Document created successfully", "id": decoded_id}

    except PyMongoError as e:
        logger.error(f"Database error during document creation: {e}")
        raise HTTPException(status_code=500, detail="Database error") from e
    except Exception as e:
        logger.error(f"Unexpected error during document creation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.delete("/document/{id}", status_code=204)
async def delete_document(id: str) -> Response:
    """
    Удаляет документ по его идентификатору.

    Args:
        id (str): Идентификатор документа (может содержать '#24' вместо '$').

    Returns:
        Response: Пустой ответ с кодом 204 при успехе.

    Raises:
        HTTPException: 404 если документ не найден.
        HTTPException: 500 при ошибке базы данных.
    """
    decoded_id = decode_id(id)
    logger.info(f"Deleting document with id: {decoded_id}")

    try:
        result = collection.delete_one({"id": decoded_id})
        if result.deleted_count == 0:
            logger.warning(f"Document with id '{decoded_id}' not found for deletion")
            raise HTTPException(status_code=404, detail="Document not found")

        logger.info(f"Document with id '{decoded_id}' deleted successfully")
        return Response(status_code=204)

    except PyMongoError as e:
        logger.error(f"Database error during document deletion: {e}")
        raise HTTPException(status_code=500, detail="Database error") from e
    except Exception as e:
        logger.error(f"Unexpected error during document deletion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


