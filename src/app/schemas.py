"""
Модели Pydantic для приложения Service Desk.

Этот модуль содержит все модели данных и схемы, используемые
во всем приложении.
"""

from pydantic import BaseModel
from typing import Optional, Dict
from fastapi import FastAPI, WebSocket
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Модель запроса для операций чата."""

    user_id: str
    chat_id: str
    chat: Optional[str] = None
    state: Optional[str] = None


class UserState:
    """Управление состоянием пользователя для сессий чата."""

    def __init__(self):
        self.chat_history = []
        self.current_state = "baseState"
        # Уточняющие состояния
        self.expecting_clarification: bool = False
        self.initial_query_for_clarification: Optional[str] = None
        self.clarification_count: int = 0  # сколько раз уже спрашивали уточнение

    def add_message(self, role: str, content: str):
        """Добавить сообщение в историю чата."""
        self.chat_history.append({"role": role, "content": content})

    def update_state(self, new_state: str):
        """Обновить текущее состояние пользователя."""
        self.current_state = new_state


class ConnectionManager:
    """
    Управляет WebSocket соединениями и состояниями пользователей.

    Этот класс обрабатывает жизненный цикл WebSocket соединений,
    поддерживает состояние пользователей между сессиями и предоставляет методы
    для отправки сообщений конкретным пользователям.
    """

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_states: Dict[str, UserState] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        """Принять WebSocket соединение и инициализировать состояние пользователя."""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        if user_id not in self.user_states:
            self.user_states[user_id] = UserState()
        logger.info(
            f"User {user_id} connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, user_id: str):
        """Удалить соединение пользователя и очистить состояние."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_states:
            del self.user_states[user_id]
        logger.info(
            f"User {user_id} disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_personal_message(self, message: dict, user_id: str):
        """Отправить сообщение конкретному пользователю через WebSocket."""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
                self.disconnect(user_id)

    def get_user_state(self, user_id: str) -> Optional[UserState]:
        """Получить состояние пользователя по заданному ID пользователя."""
        return self.user_states.get(user_id)