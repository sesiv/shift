# pytest=9.0.1
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.integration
def test_first_message_via_streamlit():
    """
    Интеграционный сценарий: имитируем первое сообщение в чат через Streamlit
    и печатаем ответ сервера без каких-либо проверок.
    """
    # Указываем куда стучаться Streamlit-у (в докере сервис проброшен на localhost:8000)
    os.environ.setdefault("FASTAPI_HOST", "localhost")

    app_path = Path(__file__).resolve().parents[2] / "app" / "web.py"
    message_text = "у меня сломался ноутбук"

    app = AppTest.from_file(str(app_path), default_timeout=60)

    # Первый прогон для инициализации стейта
    app.run()

    # Отправляем первое сообщение пользователя
    app.text_input("text_input_0").set_value(message_text)
    app.run()

    # Достаём последний ответ ассистента из state
    response_buttons = None
    state = app.session_state
    if "current_chat" in state and "chats" in state:
        current_chat = state["current_chat"]
        chats = state["chats"]
        history = chats.get(current_chat, [])
        assistant_messages = [m for m in history if m.get("role") == "assistant"]
        if assistant_messages:
            response = assistant_messages[-1]
            response_buttons = response.get("buttons")

    assert response_buttons is not None, "Ответ сервера не содержит кнопок"
    assert isinstance(response_buttons, list), "Кнопки должны быть списком"
    for btn in response_buttons:
        assert "label" in btn, "У кнопки отсутствует поле label"

    labels = [btn["label"] for btn in response_buttons]
    assert (
        "Рабочее место пользователя IT/Поддержка оборудования рабочих мест/" in labels
    ), "Нет кнопки с ожидаемым label для категории оборудования рабочих мест"
    assert (
        "Сброс" in labels or "Cброс" in labels
    ), "Нет кнопки сброса"
