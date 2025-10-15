import glob
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import requests
import json
import logging
import os
from typing import Optional, Dict, List, Tuple
import bisect
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

logging.info(os.environ.get("MONGO_URL"))
MONGO_URL = os.environ.get("MONGO_URL", "http://127.0.0.1:5017")
E5_URL = os.environ.get("E5_URL", "http://e5:5003")
VECTOR_DB_URL = os.environ.get("VECTOR_DB_URL", "http://vector_db:5004")
QUESTION_MODEL_URL = os.environ.get("QUESTION_MODEL_URL", "http://question_model:5005")
SERVER_URL = os.environ.get("SERVER_URL", "http://server:5002")

app = FastAPI()


# User state management
class UserState:
    def __init__(self):
        self.chat_history = []
        self.current_state = "baseState"
        # Clarification flow state
        self.expecting_clarification: bool = False
        self.initial_query_for_clarification: Optional[str] = None
        self.last_model_question: Optional[str] = None
        self.clarification_count: int = 0  # сколько раз уже спрашивали уточнение

    def add_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

    def update_state(self, new_state: str):
        self.current_state = new_state


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_states: Dict[str, UserState] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        if user_id not in self.user_states:
            self.user_states[user_id] = UserState()
        logging.info(f"User {user_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_states:
            del self.user_states[user_id]
        logging.info(f"User {user_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except Exception as e:
                logging.error(f"Error sending message to user {user_id}: {e}")
                self.disconnect(user_id)

    def get_user_state(self, user_id: str) -> Optional[UserState]:
        return self.user_states.get(user_id)


manager = ConnectionManager()


class ChatRequest(BaseModel):
    user_id: str
    chat_id: str
    chat: Optional[str] = None
    state: Optional[str] = None


logging.basicConfig(level=logging.INFO)

THRESHOLD = 0.15

# Calibration mapping (distance -> P(correct_top1)) loaded once at startup
CALIBRATION_PATH = os.path.join("data","calibration.json")
GLOBAL_CALIBRATION: List[Tuple[float, float]] = []


def load_global_calibration() -> List[Tuple[float, float]]:
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            cal = json.load(f)
        data = cal.get("global_iso", Exception).get("data",Exception)
        # Ensure sorted by distance ascending
        
        
        logging.info("SUCCESS LOADED calibration.json")
        logging.info(f"sorted_pairs {data}")
        return data

    except Exception as e:
        logging.error(f"Failed to load calibration mapping from {CALIBRATION_PATH}: {e}")
        # Fallback to a coarse default mapping if file missing
        return [
            (0.0, 1.0),
            (0.05, 0.9),
            (0.1, 0.8),
            (0.15, 0.7),
            (0.2, 0.6),
            (0.3, 0.5),
        ]


GLOBAL_CALIBRATION = load_global_calibration()


def distance_to_confidence(distance_value: float) -> float:
    if not GLOBAL_CALIBRATION:


        
        return 0.0
    distances = [d for d, _ in GLOBAL_CALIBRATION]
    idx = bisect.bisect_left(distances, distance_value)
    if idx <= 0:
        return GLOBAL_CALIBRATION[0][1]
    if idx >= len(GLOBAL_CALIBRATION):
        return GLOBAL_CALIBRATION[-1][1]
    d1, p1 = GLOBAL_CALIBRATION[idx - 1]
    d2, p2 = GLOBAL_CALIBRATION[idx]
    if d2 == d1:
        return p1
    # Linear interpolation between points
    alpha = (distance_value - d1) / (d2 - d1)
    return p1 + alpha * (p2 - p1)


@app.websocket("/ws/chat/{user_id}")
async def chat(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            logging.info(f"Received from user {user_id}: {data}")

            # Handle different message types
            if "message" in data:
                await handle_user_message(user_id, data["message"])
            elif "button" in data:
                await handle_button_click(user_id, data["button"])
            else:
                await manager.send_personal_message({
                    "error": "Invalid message format. Expected 'message' or 'button' field."
                }, user_id)

    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logging.error(f"Error in WebSocket connection for user {user_id}: {e}")
        manager.disconnect(user_id)


async def handle_user_message(user_id: str, message: str):
    """Handle user message through WebSocket"""
    user_state = manager.get_user_state(user_id)
    if not user_state:
        await manager.send_personal_message({"error": "User state not found"}, user_id)
        return

    # Add user message to history
    user_state.add_message("user", message)

    # If expecting clarification answer, combine with initial question and model question
    if getattr(user_state, "expecting_clarification", False) and getattr(user_state, "initial_query_for_clarification", None):
        combined_message = (
            f"Вопрос пользователя: {user_state.initial_query_for_clarification}\n"
            f"Уточняющий вопрос: {user_state.last_model_question or ''}\n"
            f"Ответ пользователя: {message}"
        )
        agg = aggregate_nodes(user_state.current_state, combined_message)
        # reset clarification flags
        user_state.expecting_clarification = False
        user_state.last_model_question = None
        # держим initial_query_for_clarification для контекста, но больше не спрашиваем новый вопрос
    else:
        # Process message using existing logic
        agg = aggregate_nodes(user_state.current_state, message)
    logging.info(f"Aggregate result: {agg}")
    
    predicted_id = agg.get("predicted_id")
    confidence = agg.get("confidence", 0.0)
    top_categories = agg.get("top_categories", [])
    logging.info(f"confidence {confidence}")
    try:
        if confidence >= 0.83 and predicted_id:
            # High confidence: fetch and return top document, plus two buttons
            doc = requests.get(
                f"{MONGO_URL}/document/{predicted_id}",
                params={"filter": "guide,description,name_path"}
            ).json()["data"]
            logging.info(f"Retrieved document: {doc}")

            if doc["guide"] != "":
                answer = doc["guide"]
            else:

                answer = "Описание категории:\n\n" + doc["description"]+f"""
                \n\n **Рекомендуем вам оформить** {(doc.get('name_path', '') or '').replace('/', '\n\n ->')[:-3]}
                \nНажмите для подтверждения
                """
                #убираем последний ->

            buttons = [
                {"label": f"Подтвердить", "value": predicted_id},
                {"label": "Это мне не подходит", "value": "no_match"},
            ]

            user_state.add_message("assistant", answer)
            await manager.send_personal_message({
                "text": answer,
                "type": "message_response",
                "new_state": None,
                "predicted_id": predicted_id,
                "confidence": confidence,
                "buttons": buttons,
            }, user_id)

        elif 0.5 <= confidence < 0.83 or user_state.clarification_count >= 1:
            # Medium confidence: suggest top-5 categories as buttons
            suggestion_buttons = []
            for item in top_categories[:5]:
                cid = item["id"]
                
                try:
                    cdoc = requests.get(
                        f"{MONGO_URL}/document/{cid}",
                        params={"filter": "name_path"}
                    ).json()["data"]
                    label = cdoc.get("name_path")
                    logging.info(label.split("/"))
                    label_cropped = "/".join(str(label).split("/")[2:])
                    logging.info(f"Button {label_cropped}")
                except Exception:
                    label = cid
                suggestion_buttons.append({"label": label_cropped, "value": f"open_doc:{cid}"})

            prompt_text = "Выберите наиболее подходящую категорию работ"
            user_state.add_message("assistant", prompt_text)
            logging.info(f"Button {suggestion_buttons}")
            await manager.send_personal_message({
                "text": prompt_text,
                "type": "message_response",
                "new_state": None,
                "predicted_id": predicted_id,
                "confidence": confidence,
                "buttons": suggestion_buttons,
            }, user_id)

        else:
            # Low confidence: ask a clarifying question using question_model
            candidate_ids = [c["id"] for c in top_categories[:5]]
            question_text = generate_clarifying_question(message, candidate_ids)
            user_state.expecting_clarification = True
            if not user_state.initial_query_for_clarification:
                user_state.initial_query_for_clarification = message
            user_state.last_model_question = question_text
            user_state.clarification_count += 1
            user_state.add_message("assistant", question_text)
            await manager.send_personal_message({
                "text": question_text,
                "type": "message_response",
                "new_state": None,
                "predicted_id": predicted_id,
                "confidence": confidence,
                "buttons": [],
            }, user_id)

    except Exception as e:
        logging.error(f"Error processing message for user {user_id}: {e}")
        await manager.send_personal_message({
            "error": "Ошибка при обработке сообщения"
        }, user_id)


async def handle_button_click(user_id: str, button: str):
    """Handle button click through WebSocket"""
    user_state = manager.get_user_state(user_id)
    if not user_state:
        await manager.send_personal_message({"error": "User state not found"}, user_id)
        return

    logging.info(f"Button clicked by user {user_id}: {button}")

    # Special handling for negative feedback
    if button == "no_match":
        answer = "Извините, что не нашли нужный вариант. Опишите, пожалуйста, задачу другими словами."
        new_state = manager.get_user_state(user_id).current_state if manager.get_user_state(user_id) else "baseState"
    else:
        # If the button asks to open a document (from medium-confidence suggestions)
        if isinstance(button, str) and button.startswith("open_doc:"):
            node_id = button.split(":", 1)[1]
            try:
                doc = requests.get(
                    f"{MONGO_URL}/document/{node_id}",
                    params={"filter": "guide,description,name_path"}
                ).json()["data"]
                
                if doc["guide"] != "":
                    answer = doc["guide"]
                else:

                    answer = "Описание категории:\n\n" + doc["description"]+f"""
                    \n\n **Рекомендуем вам оформить** {(doc.get('name_path', '') or '').replace('/', '\n\n ->')[:-3]}
                    \nНажмите для подтверждения
                    """
                    
                buttons = [
                    {"label": f"Подтвердить", "value": node_id},
                    {"label": "Это мне не подходит", "value": "no_match"},
                ]


                user_state.update_state(node_id)
                user_state.add_message("assistant", answer)
                await manager.send_personal_message({
                    "text": answer,
                    "type": "button_response",
                    "new_state": node_id,
                    "buttons": buttons,
                }, user_id)
                return
            except Exception as e:
                logging.error(f"Error fetching document for node {node_id}: {e}")
                await manager.send_personal_message({"error": "Ошибка при получении документа"}, user_id)
                return

        # Otherwise treat as a normal node id click
        # Get children nodes for the button
        children = get_children(button)

        if children:
            # Generate question for children
            question = get_question(button)
            answer = question
            new_state = button
        else:
            # Leaf node selected – ask for rating 1..10
            answer = "Спасибо оцените работу"
            new_state = button

    # Update user state
    user_state.update_state(new_state)
    user_state.add_message("assistant", answer)

    # Send response to client
    await manager.send_personal_message({
        "text": answer,
        "type": "button_response",
        "new_state": new_state
    }, user_id)


# обработчик сохранения чата (сохраняем для совместимости)
@app.post("/save_chat")
async def save_chat(chat_data: ChatRequest):
    payload = {
        "user_id": chat_data.user_id,
        "chat_id": chat_data.chat_id,
        "chat": chat_data.chat,
        "state": chat_data.state
    }
    response = requests.post(f"{SERVER_URL}/save_chat", json=payload)
    return response.json()["message"]


# Вспомогательные функции
# классификатор сообщений
def aggregate_nodes(state: str, message: str) -> dict:
    """
    Классифицирует сообщение в один из узлов и вычисляет откалиброванную уверенность.

    Возвращает словарь с ключами:
    - predicted_id: str | None — предсказанный идентификатор узла
    - confidence: float в диапазоне [0,1] — уверенность в предсказании
    - top_categories: список словарей {id, score}, отсортированных по убыванию score (до всех доступных)
    - best_distance: float | None — минимальная дистанция для предсказанной категории
    """
    message_vector = get_vector(message)
    similar_nodes = search_similar_nodes(state, message_vector)
    similar_nodes_dict = json.loads(similar_nodes.get("response", "[]"))
    logging.info(f"similar_nodes_dict {similar_nodes_dict}")

    hits = {"folder": {}, "slmService": {}, "categoriesWork": {}}

    if state == "baseState":
        for node in similar_nodes_dict:
            hits["folder"][node["folder"]] = hits["folder"].get(node["folder"], 0) + node["distance"]
            hits["slmService"][node["slmService"]] = hits["slmService"].get(node["slmService"], 0) + node["distance"]
            hits["categoriesWork"][node["categoriesWork"]] = hits["categoriesWork"].get(node["categoriesWork"], 0) + node["distance"]
    elif state == "folder":
        for node in similar_nodes_dict:
            hits["slmService"][node["slmService"]] = hits["slmService"].get(node["slmService"], 0) + node["distance"]
            hits["categoriesWork"][node["categoriesWork"]] = hits["categoriesWork"].get(node["categoriesWork"], 0) + node["distance"]
    elif state == "slmService":
        for node in similar_nodes_dict:
            hits["categoriesWork"][node["categoriesWork"]] = hits["categoriesWork"].get(node["categoriesWork"], 0) + node["distance"]

    logging.info(f"hits {hits}")

    best = {"folder": {"id": "", "score": 0.0}, "slmService": {"id": "", "score": 0.0},
            "categoriesWork": {"id": "", "score": 0.0}}

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
        if abs(best[priority]["score"] - max_score) < 0.01:
            predicted_id = best[priority]["id"]
            break

    # Определяем минимальную дистанцию для предсказанной категории для калибровки уверенности
    best_distance = None
    if predicted_id:
        candidate_distances: List[float] = []
        # Собираем дистанции для нод, у которых совпадает predicted_id на любом уровне
        for node in similar_nodes_dict:
            if node.get("categoriesWork") == predicted_id or node.get("slmService") == predicted_id or node.get("folder") == predicted_id:
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

    confidence = distance_to_confidence(best_distance) if best_distance is not None else 0.0

    # Формируем топ категорий (по убыванию агрегированного score) для подсказок
    categories_scores = hits["categoriesWork"]
    sorted_categories = sorted(categories_scores.items(), key=lambda x: x[1], reverse=True)
    top_categories = [{"id": cid, "score": score} for cid, score in sorted_categories]

    result = {
        "predicted_id": predicted_id,
        "confidence": confidence,
        "top_categories": top_categories,
        "best_distance": best_distance,
    }
    return result


SERVICE_PATH = os.path.join("data","services.json")

# получение детей текущей ноды
def get_children(state: str):
    logging.info(f"Текущая директория: {os.getcwd()}")
    with open(SERVICE_PATH, 'r', encoding='utf-8') as f:
        node_map = json.load(f)

    categories_name = []
    logging.info(f"state: {state}")
    for children_id in node_map[state]["children_id"]:
        categories_name.append(node_map[children_id]["name"])

    return categories_name


def get_node_name(node_id: str) -> Optional[str]:
    try:
        with open(SERVICE_PATH, 'r', encoding='utf-8') as f:
            node_map = json.load(f)
        node = node_map.get(node_id)
        if node:
            return node.get("name")
    except Exception:
        pass
    return None


# Интеграция с другими сервисами
# векторизация сообщения
def get_vector(text: str):
    response = requests.post(f"{E5_URL}/get_vector", json={"query": text})
    return response.json()["vector"]


# поиск ближайшей ноды
def search_similar_nodes(state, vector):
    response = requests.post(f"{VECTOR_DB_URL}/ticket/search", json={"state": state,
                                                                   "query_vector": vector})
    return response.json()


# получение уточняющего вопроса
def get_question(node: str, categories=None):
    if categories is None:
        categories = get_children(node)
    try:
        response = requests.post(f"{QUESTION_MODEL_URL}/generate-question", json={"categories": categories})
        return response.json()["choices"][0]["message"]['content']
    except Exception as e:
        logging.error(f"Error generating question: {e}")
        return "Ошибка при генерации вопроса"


def generate_clarifying_question(user_question: str, category_ids: List[str]) -> str:
    """Call question_model to generate a clarifying question.

    We pass human-readable category labels (name_path) and the original user question.
    """
    try:
        labels: List[str] = []
        for cid in category_ids:
            try:
                cdoc = requests.get(
                    f"{MONGO_URL}/document/{cid}",
                    params={"filter": "name_path"}
                ).json()["data"]
                labels.append(cdoc.get("name_path", cid))
            except Exception:
                labels.append(cid)

        payload = {"categories": labels, "question": user_question}
        response = requests.post(f"{QUESTION_MODEL_URL}/generate-question", json=payload)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Error generating clarifying question: {e}")
        return "Не совсем понял. Уточните, пожалуйста, задачу одним предложением."
