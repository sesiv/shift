from fastapi.responses import Response
from pymongo import MongoClient
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel , Field
from typing import List
import logging
import os

logging.basicConfig(level=logging.INFO)

MONGO_URI = os.getenv("MONGO_URI")
logging.info(f"Connecting to MongoDB at {MONGO_URI}")
try:
    client = MongoClient(MONGO_URI)
    client.server_info()
    logging.info("Successfully connected to MongoDB.")
except Exception as e:
    logging.error(f"FATAL: Failed to connect to MongoDB: {e}")
    raise

db = client.get_database("nodes")

collection = db.get_collection("nodes")

app = FastAPI()


class Node(BaseModel):
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



@app.get("/document/{id}")
async def getDocument(id, filter: str | None = Query(
    default=None,
    description="""
    Список полей которые нужно передать через запятую

    Если пустой то передаем весь обьект 
    """
)):
    id = id.replace(r"#24",
                    "$")  # во время пересыла $ декодируется как специальный символ а как его экранировать я не разобрался

    logging.info(f"id {id}")
    try:
        node_from_mongo = collection.find_one({"id": id})  # type dict
        logging.info(f"node_from_mongo {node_from_mongo}")
        node_from_mongo.pop("_id")


    except:
        logging.info("status404 node not found")
        return {"status": "404 node not found "}

    node = Node(**node_from_mongo)  # type Node(BaseModel)

    logging.info(f"node.__dict__ {node.__dict__}")
    if filter:
        include_set = set(filter.split(","))
        return {"status": 200, "data": node.model_dump(include=include_set)}

    return {"status": 200, "data": node}


@app.post("/document/{id}", status_code=201)
async def createDocument(id: str, node: Node):
    """
    Создает новый документ с указанным id
    """
    id = id.replace(r"#24", "$")

    logging.info(f"Creating document with id: {id}")

    # Проверяем, существует ли уже документ с таким id
    existing_node = collection.find_one({"id": id})
    if existing_node:
        logging.info(f"Document with id {id} already exists")
        raise HTTPException(status_code=409, detail="Document with this id already exists")

    try:
        node_dict = node.model_dump()
        node_dict["id"] = id  # Убеждаемся, что id соответствует пути

        result = collection.insert_one(node_dict)

        logging.info(f"Document created successfully with MongoDB _id: {result.inserted_id}")
        return {"message": "Document created successfully", "id": id}

    except Exception as e:
        logging.error(f"Error creating document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.delete("/document/{id}")
async def deleteDocument(id: str):
    """
    Удаляет документ с указанным id
    """
    id = id.replace(r"#24", "$")

    logging.info(f"Deleting document with id: {id}")

    try:
        result = collection.delete_one({"id": id})

        if result.deleted_count == 0:
            logging.info(f"Document with id {id} not found")
            raise HTTPException(status_code=404, detail="Document not found")
        logging.info(f"Document with id {id} deleted successfully")
        return Response(status_code=204)

    except Exception as e:
        logging.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
