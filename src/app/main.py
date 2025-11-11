"""
Основное приложение Service Desk

Это основное FastAPI приложение, которое обрабатывает WebSocket соединения,
управление состоянием пользователей и взаимодействие в чате для системы Service Desk.
"""

import logging
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Импорт локальных модулей
from schemas import ChatRequest, ConnectionManager
from consts import  (
    CONFIDENCE_CONSTANTS,
    MONGO_URL, SERVER_URL,
    QUESTION_MODEL_URL, VECTOR_DB_URL,
    CLARIFICATION_COUNT_TRESHOLD
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="Service Desk API",
    description="Сервис чата в реальном времени для Service Desk с интеллектуальной маршрутизацией",
)

# Глобальный экземпляр менеджера соединений
manager = ConnectionManager()


@app.websocket("/ws/chat/{user_id}")
async def chat(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint для коммуникации чата в реальном времени.

    Обрабатывает входящие сообщения и нажатия кнопок от пользователей,
    поддерживает состояние разговора и предоставляет интеллектуальные ответы.
    """
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_json()
            logger.info(f"Received from user {user_id}: {data}")

            # Маршрутизация сообщения по типу
            if "message" in data:
                await handle_user_message(user_id, data["message"])
            elif "button" in data:
                await handle_button_click(user_id, data["button"])
            else:
                await manager.send_personal_message(
                    {
                        "error": "Invalid message format. Expected 'message' or 'button' field."
                    },
                    user_id,
                )

    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"Error in WebSocket connection for user {user_id}: {e}")
        manager.disconnect(user_id)


async def handle_user_message(user_id: str, message: str):
    """
    Обработать сообщение пользователя и сгенерировать соответствующий ответ.

    Эта функция обрабатывает основную логику обработки сообщений:
    - Добавляет сообщение в историю чата
    - Обрабатывает сценарии уточнения
    - Маршрутизирует к соответствующим категориям услуг
    - Генерирует ответы на основе уровней уверенности
    """
    user_state = manager.get_user_state(user_id)
    if not user_state:
        await manager.send_personal_message({"error": "User state not found"}, user_id)
        return

    # Добавить сообщение пользователя в историю
    user_state.add_message("user", message)
    logger.info(f"User {user_id} message: {message}")

    # Обработать сценарии уточнения
    if (
        user_state.expecting_clarification
    ):
        all_previous_user_messages = " ".join(msg["content"] for msg in user_state.chat_history if msg["role"] == "user")
        message = all_previous_user_messages + " " + message
        # Очистить состояние уточнения
        user_state.expecting_clarification = False

    agg = requests.post(f"{VECTOR_DB_URL}/aggregate",
                        params={
                              "state": user_state.current_state,
                              "message": message}
                        ).json()

    logger.info(f"Aggregate result: {agg}")

    predicted_id = agg.get("predicted_id")
    confidence = agg.get("confidence", 0.0)
    top_categories = agg.get("top_categories", [])
    logger.info(f"confidence {confidence}")

    try:
        if confidence >= CONFIDENCE_CONSTANTS[0] and predicted_id:
            # Высокая уверенность: предоставить прямой ответ
            await _handle_high_confidence_response(user_id, predicted_id)

        elif (
            CONFIDENCE_CONSTANTS[-1] <= confidence < CONFIDENCE_CONSTANTS[0]
            or user_state.clarification_count >= CLARIFICATION_COUNT_TRESHOLD
        ):
            # Средняя уверенность: предложить топ категории
            await _handle_medium_confidence_response(
                user_id, top_categories, predicted_id
            )

        else:
            # Низкая уверенность: сгенерировать уточняющий вопрос
            await _handle_low_confidence_response(
                user_id, message, top_categories, predicted_id
            )

    except Exception as e:
        logger.error(f"Error processing message for user {user_id}: {e}")
        await manager.send_personal_message(
            {"error": "Ошибка при обработке сообщения"}, user_id
        )


async def _handle_high_confidence_response(
    user_id: str, predicted_id: str
):
    """Обрабатывает ответы с высокой уверенностью с выводом
     прямых ответов на вопрос."""
    user_state = manager.get_user_state(user_id)

    doc = requests.get(
        f"{MONGO_URL}/document/{predicted_id}",
        params={"filter": "guide,description,name_path"},
    ).json()["data"]
    logger.info(f"Retrieved document: {doc}")

    if doc["guide"] != "":
        answer = doc["guide"]
    else:
        answer = (
            "Описание категории:\n\n"
            + doc["description"]
            + f"""
            \n\n **Рекомендуем вам оформить** {(doc.get("name_path", "") or "").replace("/", "\n\n ->")[:-3]}
            \nНажмите для подтверждения
            """
        )

    buttons = [
        {"label": "Подтвердить", "value": predicted_id},
        {"label": "Это мне не подходит", "value": "no_match"},
    ]

    user_state.add_message("assistant", answer)
    await manager.send_personal_message(
        {
            "text": answer,
            "type": "message_response",
            "new_state": None,
            "predicted_id": predicted_id,
            "buttons": buttons,
        },
        user_id,
    )


async def _handle_medium_confidence_response(
    user_id: str, top_categories: list, predicted_id: str
):
    """Обрабатывает ответы со средней уверенностью с выводом
     возможных категорий."""
    user_state = manager.get_user_state(user_id)

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
            label_cropped = cid
        suggestion_buttons.append({
            "label": label_cropped,
            "value": f"open_doc:{cid}",
        })

    # Добавляем кнопку "Сброс"
    suggestion_buttons.append({"label": "Сброс", "value": "no_categories"})

    prompt_text = "Выберите наиболее подходящую категорию работ"
    user_state.add_message("assistant", prompt_text)
    logger.info(f"Button suggestions: {suggestion_buttons}")
    await manager.send_personal_message(
        {
            "text": prompt_text,
            "type": "message_response",
            "new_state": None,
            "predicted_id": predicted_id,
            "buttons": suggestion_buttons,
        },
        user_id,
    )


async def _handle_low_confidence_response(
    user_id: str,
    message: str,
    top_categories: list,
    predicted_id: str,
):
    """Обрабатывает ответы с низкой уверенностью с выводом уточняющего вопроса."""
    user_state = manager.get_user_state(user_id)

    candidate_ids = [c["id"] for c in top_categories[:5]]
    
    try:
        question_text = requests.post(
                f"{QUESTION_MODEL_URL}/generate-question", json={"categories": candidate_ids, "question": message}
            ).json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Error generating question: {e}")
        question_text="Ошибка при генерации вопроса"

    user_state.expecting_clarification = True
    if not user_state.initial_query_for_clarification:
        user_state.initial_query_for_clarification = message
    user_state.clarification_count += 1
    user_state.add_message("assistant", question_text)
    await manager.send_personal_message(
        {
            "text": question_text,
            "type": "message_response",
            "new_state": None,
            "predicted_id": predicted_id,
            "buttons": [],
        },
        user_id,
    )


async def handle_button_click(user_id: str, button: str):
    """
    Обрабатывает различные типы взаимодействий с кнопками
    от пользователей:
    - Открытие документов
    - Навигация по категориям
    - Подтверждение/отклонение
    """
    user_state = manager.get_user_state(user_id)
    if not user_state:
        await manager.send_personal_message({"error": "User state not found"}, user_id)
        return

    logger.info(f"Button clicked by user {user_id}: {button}")
    logger.info(f"Current state user {user_id}: {user_state.current_state}")

    if button == "no_match":
        # Обработать ответ "не подходит"
        answer = "Извините, что не нашли нужный вариант. Опишите, пожалуйста, задачу другими словами."
        new_state = (
            manager.get_user_state(user_id).current_state
            if manager.get_user_state(user_id)
            else "baseState"
        )

    elif button == "no_categories":
        # Сбрасываем состояние до базового и увеличиваем счетчик уточнений
        new_state = "baseState"
        user_state.clarification_count += 1  # Увеличиваем счетчик уточнений
        answer = "Извините, что не нашли нужный вариант. Опишите, пожалуйста, задачу другими словами."
        logger.info(
            f"Processing 'no_categories' button: answer='{answer}', new_state='{new_state}'"
        )
    else:
        if isinstance(button, str) and button.startswith("open_doc:"):
            # Обработать открытие документа
            node_id = button.split(":", 1)[1]
            try:
                doc = requests.get(
                    f"{MONGO_URL}/document/{node_id}",
                    params={"filter": "guide,description,name_path"},
                ).json()["data"]

                if doc["guide"] != "":
                    answer = doc["guide"]
                else:
                    answer = (
                        "Описание категории:\n\n"
                        + doc["description"]
                        + f"""
                        \n\n **Рекомендуем вам оформить** {(doc.get("name_path", "") or "").replace("/", "\n\n ->")[:-3]}
                        \nНажмите для подтверждения
                        """
                    )

                buttons = [
                    {"label": "Подтвердить", "value": node_id},
                    {"label": "Это мне не подходит", "value": "no_match"},
                ]

                user_state.update_state(node_id)
                user_state.add_message("assistant", answer)
                await manager.send_personal_message(
                    {
                        "text": answer,
                        "type": "button_response",
                        "new_state": node_id,
                        "buttons": buttons,
                    },
                    user_id,
                )
                return
            except Exception as e:
                logger.error(f"Error fetching document for node {node_id}: {e}")
                await manager.send_personal_message(
                    {"error": "Ошибка при получении документа"}, user_id
                )
                return
        else:
            answer = "Спасибо оцените работу"
            new_state = button

    # Обновить состояние пользователя и отправить ответ
    user_state.update_state(new_state)
    user_state.add_message("assistant", answer)
    await manager.send_personal_message(
        {"text": answer, "type": "button_response", "new_state": new_state}, user_id
    )


@app.post("/save_chat")
async def save_chat(chat_data: ChatRequest):
    """
    Сохраняет данные чата в базe путем
    пересылки данных чата в сервис сервера(server.py).
    """
    payload = {
        "user_id": chat_data.user_id,
        "chat_id": chat_data.chat_id,
        "chat": chat_data.chat,
        "state": chat_data.state,
    }
    response = requests.post(f"{SERVER_URL}/save_chat", json=payload)
    return response.json()["message"]
