from fastapi.responses import Response
from pymongo import MongoClient
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import logging
import os

logging.basicConfig(level=logging.INFO)
#ssh -fN -g -L 27018:localhost:27017 root@195.133.66.35 новый проброс для mongo.py в докере
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

""""
пример ноды какая должна быть 
{'_id': ObjectId('688f0b261c0bd9517c9460a9'), - не трогать это сама монго делает 
 'id': 'categoriesWork$48859506',
 'sd_level': 'categoriesWork',
 'description': 'Категория работ "Открытие/изменение овердрафта" включает в себя обращения пользователей, связанные с техническими и организационными вопросами, возникающими при работе с сервис-деском, а также с процессами, связанными с открытием, изменением и согласованием заявок в рамках овердрафта. Категория охватывает проблемы с интерфейсом сервис-деска (недоступность элементов, ошибки при создании заявок, некорректное отображение данных), вопросы маршрутизации заявок, согласования доступов, настройки прав, а также запросы на изменение структуры данных, корректировку процессов и уточнение процедур. Также в нее входят обращения по вопросам доступа к ресурсам, настройки компонентов, проверки корректности заполнения данных в системе, а также временные изменения в процессе согласования в связи с отпуском сотрудников.',
 'guide': '### Руководство: Открытие или изменение овердрафта\n\nЧтобы оформить или изменить условия овердрафта по вашей зарплатной карте, пожалуйста, следуйте этим шагам:\n\n1.  **Заполните заявление.** Вам необходимо заполнить «Заявление на предоставление овердрафта». Эту форму можно скачать на корпоративном портале в разделе «Финансы» -> «Шаблоны документов» или запросить прямо у меня, написав «скачать заявление на овердрафт».\n\n2.  **Укажите детали.** В заявлении обязательно укажите желаемый лимит овердрафта и срок, на который он требуется.\n\n3.  **Отправьте на согласование.** После заполнения создайте заявку в Service Desk через этого бота, выбрав категорию «Открытие/изменение овердрафта», и приложите скан-копию подписанного заявления. Ваша заявка будет автоматически направлена на согласование вашему руководителю, а затем в финансовый отдел.\n\n4.  **Ожидайте уведомления.** Вы получите уведомление об изменении статуса вашей заявки и о финальном решении. Стандартный срок рассмотрения — 3 рабочих дня.',
 'children': [],
 'path': '/folder$1115801/slmService$1116002/categoriesWork$48859506/',
 'name_path': '/02. Социальная программа, кадровые вопросы, оплата труда, справки/Финансы (Справки о доходах, отпускные, расчетный лист, овердрафт)/Открытие/изменение овердрафта/',
 'name': 'Открытие/изменение овердрафта'}

"""


class Node(BaseModel):
    id: str
    sd_level: str  # можно было поставить Literal но непонятно что может добавляться в будущем
    description: str
    guide: str
    children: list
    path: str
    name_path: str
    name: str


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
