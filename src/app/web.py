"""
Модуль отвечает за фронтенд приложения
"""

import streamlit as st
import websocket
import uuid
import json
import os
import logging
import base64
import requests


logging.basicConfig(level=logging.INFO)

# Широкий макет страницы для уменьшения боковых полей
st.set_page_config(layout="wide")

# Конфигурация URL

FASTAPI_HOST = os.getenv("FASTAPI_HOST", "main")

# Базовые URL
fastapi_base_url = f"http://{FASTAPI_HOST}:8000"
fastapi_ws_base_url = f"ws://{FASTAPI_HOST}:8000"

# URL для сервисов
fastapi_audio_url = f"http://asr:5006/transcribe"
fastapi_save_chat_url = f"{fastapi_base_url}/save_chat"  # Для совместимости

# Инициализация состояния сессии
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "chats" not in st.session_state:
    st.session_state.chats = {"Чатик 1": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Чатик 1"
if "num_chat" not in st.session_state:
    st.session_state.num_chat = 1
if "audio_recorder_key" not in st.session_state:
    st.session_state.audio_recorder_key = 0
if "ws" not in st.session_state:
    st.session_state.ws = None

# Инициализация первого сообщения, если чат пуст
if not st.session_state.chats[st.session_state.current_chat]:
    st.session_state.chats[st.session_state.current_chat].append({
        "role": "assistant",
        "content": "Здравствуйте! Как я могу вам помочь?"
    })

st.title("Чатик " + st.session_state.current_chat.split()[-1])


# Глобальные стили (увеличение размера шрифта)
def _inject_font_styles():
    st.markdown(
        """
        <style>
        /* Увеличиваем размер шрифта в сообщениях чата */
        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            font-size: 18px; /* основной текст */
            line-height: 1.6;
        }
        div[data-testid="stChatMessage"] {
            font-size: 18px;
        }
        /* Поле ввода */
        .stTextInput input {
            font-size: 18px !important;
        }
        /* Кнопки */
        .stButton { width: 100% !important; }
        .stButton > button {
            width: 100% !important;
            font-size: 18px !important;
        }
        /* Нормализуем типографику внутри кнопок, чтобы не отличалась от других элементов */
        .stButton p, .stButton div {
            font-size: 18px !important;
            line-height: 1.4 !important;
            margin: 0.2 !important;
        }
        /* Уменьшаем горизонтальные отступы контейнера */
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_font_styles()

#Функции управления WebSocket и чатами

def get_ws_connection():
    """
    Создает и кеширует WebSocket соединение в session_state.
    Переподключается в случае разрыва.
    """
    ws_url = f"{fastapi_ws_base_url}/ws/chat/{st.session_state.user_id}"

    if st.session_state.ws is None or not st.session_state.ws.connected:
        try:
            logging.info(f"Connecting to WebSocket at {ws_url}")
            st.session_state.ws = websocket.create_connection(ws_url, timeout=60)
            logging.info("WebSocket connection established.")
        except (websocket.WebSocketException, ConnectionRefusedError, TimeoutError) as e:
            st.error(f"Не удалось подключиться к WebSocket серверу: {e}")
            st.session_state.ws = None
            return None
    return st.session_state.ws


def ws_send_and_recv(payload: dict):
    """Send a JSON payload over WS and return parsed JSON with auto-reconnect on failure."""
    ws = get_ws_connection()
    if not ws:
        return {"error": "Нет соединения с сервером"}
    try:
        ws.send(json.dumps(payload))
        resp_str = ws.recv()
        return json.loads(resp_str)
    except (websocket.WebSocketException, BrokenPipeError, OSError, json.JSONDecodeError) as e:
        logging.error(f"WS error during send/recv: {e}. Reconnecting...")
        st.session_state.ws = None
        ws = get_ws_connection()
        if not ws:
            return {"error": str(e)}
        try:
            ws.send(json.dumps(payload))
            resp_str = ws.recv()
            return json.loads(resp_str)
        except (websocket.WebSocketException, BrokenPipeError, OSError, json.JSONDecodeError) as e2:
            logging.error(f"WS retry failed: {e2}")
            st.session_state.ws = None
            return {"error": str(e2)}


def new_chat():
    st.session_state.num_chat += 1
    chat_name = f"Чатик {st.session_state.num_chat}"
    st.session_state.chats[chat_name] = [{
        "role": "assistant",
        "content": "Здравствуйте! Как я могу вам помочь?"
    }]
    st.session_state.current_chat = chat_name
    st.rerun()


def delete_chat(chat_name):
    if len(st.session_state.chats) > 1:
        current_index = list(st.session_state.chats.keys()).index(chat_name)
        del st.session_state.chats[chat_name]
        new_chat_name = list(st.session_state.chats.keys())[max(0, current_index - 1)]
        st.session_state.current_chat = new_chat_name
        st.rerun()


def save_current_chat():
    """Сохраняет чат через HTTP эндпоинт для совместимости."""
    chat_to_save = []
    for message in st.session_state.chats[st.session_state.current_chat]:
        message_copy = message.copy()
        if "audio" in message_copy and isinstance(message_copy["audio"], bytes):
            message_copy["audio"] = base64.b64encode(message_copy["audio"]).decode('utf-8')
        chat_to_save.append(message_copy)

    chat_str = json.dumps(chat_to_save, ensure_ascii=False)
    payload = {
        "user_id": st.session_state.user_id,
        "chat_id": st.session_state.current_chat,
        "chat": chat_str,
        "state": "baseState"
    }
    try:
        requests.post(fastapi_save_chat_url, json=payload)
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to save chat: {e}")


# Боковая панель
with st.sidebar:
    st.subheader("💬 Ваши чаты")
    for chat_name in list(st.session_state.chats.keys()):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(chat_name, key=f"chat_btn_{chat_name}", use_container_width=True):
                st.session_state.current_chat = chat_name
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_btn_{chat_name}", use_container_width=True):
                delete_chat(chat_name)
    if st.button("➕ Новый чат", use_container_width=True):
        new_chat()

# Отображение истории чата
messages = st.session_state.chats[st.session_state.current_chat]
for idx, message in enumerate(messages):
    with st.chat_message(message["role"]):
        if message.get("audio"):
            st.audio(message["audio"], format='audio/wav')
        if message.get("content"):
            content = message["content"]
            st.markdown(content)
            # Рендерим динамические кнопки, если они есть в сообщении ассистента
            if message["role"] == "assistant" and message.get("buttons"):
                buttons = message.get("buttons", [])
                for b_idx, btn in enumerate(buttons):
                    label = btn.get("label", "Выбрать")
                    value = btn.get("value")
                    if st.button(label, key=f"dyn_btn_{st.session_state.current_chat}_{idx}_{b_idx}", use_container_width=True):
                        resp = ws_send_and_recv({"button": value})
                        if "error" in resp:
                            ans_text = f"Ошибка сервера: {resp['error']}"
                            new_msg = {"role": "assistant", "content": ans_text}
                        else:
                            ans_text = resp.get("text", "Не удалось получить ответ от сервера.")
                            new_msg = {"role": "assistant", "content": ans_text}
                            if resp.get("buttons"):
                                new_msg["buttons"] = resp["buttons"]
                        st.session_state.chats[st.session_state.current_chat].append(new_msg)
                        save_current_chat()
                        st.rerun()
            # Render a button for the folder path if present in assistant message
            if message["role"] == "assistant" and "Полный путь до папки" in content:
                lines = content.splitlines()
                for i, line in enumerate(lines):
                    if line.strip().startswith("Полный путь до папки"):
                        if i + 1 < len(lines):
                            path_line = lines[i + 1].strip()
                            if path_line:
                                if st.button(path_line, key=f"path_btn_{st.session_state.current_chat}_{idx}", use_container_width=True):
                                    thanks_text = "Спасибо оцените работу"
                                    st.session_state.chats[st.session_state.current_chat].append({
                                        "role": "assistant",
                                        "content": thanks_text
                                    })
                                    st.rerun()
                        break

            # Render horizontal rating buttons 1..10 when asked to rate
            if message["role"] == "assistant" and content.strip() in ("Спасибо оцените работу", "Отлично, оставьте свой отзыв."):
                cols = st.columns(10)
                for rating, col in enumerate(cols, start=1):
                    with col:
                        if st.button(str(rating), key=f"rate_btn_{st.session_state.current_chat}_{idx}_{rating}"):
                            st.session_state.chats[st.session_state.current_chat].append({
                                "role": "assistant",
                                "content": f"Спасибо за оценку "
                            })
                            st.rerun()

#Обработка ввода пользователя
final_user_message = None

input_container = st.container()
with input_container:
    col1, col2 = st.columns([1, 8])
    #with col1:
        # Используем audio_recorder_key для создания уникального ключа виджета
        #recorded_audio = audiorecorder("🎤", "Запись...", key=f"audio_recorder_{st.session_state.audio_recorder_key}")
    with col2:
        text_input = st.text_input(
            "Ваше сообщение...",
            key=f"text_input_{st.session_state.audio_recorder_key}",
            label_visibility="collapsed"
        )

recorded_audio = b''  # Заглушка для аудио ввода, так как audiorecorder отключен
# Определение, есть ли новый ввод (текст или аудио)
if len(recorded_audio) > 0:
    audio_bytes = recorded_audio.export().read()
    st.session_state.chats[st.session_state.current_chat].append({
        "role": "user", "audio": audio_bytes, "content": "*(Расшифровка...)*"
    })
    with st.spinner('Расшифровываю аудио...'):
        try:
            files = {'audio_file': ('audio.wav', audio_bytes, 'audio/wav')}
            response = requests.post(fastapi_audio_url, files=files)
            response.raise_for_status()
            transcribed_text = response.json().get('text')

            final_user_message = transcribed_text
            # Обновляем последнее сообщение с расшифрованным текстом
            st.session_state.chats[st.session_state.current_chat][-1]["content"] = final_user_message
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка при отправке аудио: {e}")
            st.session_state.chats[st.session_state.current_chat][-1]["content"] = "*(Ошибка расшифровки)*"

elif text_input:
    final_user_message = text_input
    st.session_state.chats[st.session_state.current_chat].append({"role": "user", "content": final_user_message})

# Если был получен новый ввод (текстовый или голосовой), отправляем его
if final_user_message:
    with st.chat_message("assistant"):
        with st.spinner('Думаю...'):
            response_data = ws_send_and_recv({"message": final_user_message})

            if "error" in response_data:
                answer_text = f"Ошибка сервера: {response_data['error']}"
            else:
                answer_text = response_data.get("text", "Не удалось получить ответ от сервера.")

            st.markdown(answer_text)
            assistant_message = {"role": "assistant", "content": answer_text}
            if response_data.get("buttons"):
                assistant_message["buttons"] = response_data["buttons"]
            st.session_state.chats[st.session_state.current_chat].append(assistant_message)

    save_current_chat()
    # Инкрементируем ключ ПОСЛЕ полной обработки, чтобы сбросить виджет
    st.session_state.audio_recorder_key += 1
    st.rerun()