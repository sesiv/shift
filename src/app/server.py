from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from typing import Optional

app = FastAPI()

# Удаляем базу данных, если существует
if os.path.exists("chat_ratings.db"):
    os.remove("chat_ratings.db")

# Настройка базы данных
DATABASE_URL = "sqlite:///./chat_ratings.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Модель базы данных
class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(50), nullable=False)
    chat_id = Column(String(50), nullable=False)
    chat = Column(Text, nullable=False)
    state = Column(Integer, default=0)

    __table_args__ = (UniqueConstraint('user_id', 'chat_id', name='_user_chat_uc'),)


# Создание таблиц
Base.metadata.create_all(bind=engine)


# Модель запроса через Pydantic
class ChatRequest(BaseModel):
    user_id: str
    chat_id: str
    chat: Optional[str] = None
    state: Optional[str] = None

@app.post("/save_chat")
def save_chat(chat_data: ChatRequest):
    """Сохранение или обновление чата"""
    #print(chat_data.user_id, chat_data.chat_id, chat_data.chat, chat_data.state)

    db = SessionLocal()
    try:
        existing_chat = db.query(Chat).filter_by(user_id=chat_data.user_id, chat_id=chat_data.chat_id).first()

        if existing_chat:
            existing_chat.chat = chat_data.chat
            existing_chat.state = chat_data.state
            db.commit()
            return {"message": "Чат обновлен"}
        else:
            new_chat = Chat(
                user_id=chat_data.user_id,
                chat_id=chat_data.chat_id,
                chat=chat_data.chat,
                state=chat_data.state
            )
            db.add(new_chat)
            db.commit()
            return {"message": "Чат сохранен"}
    finally:
        db.close()


@app.get("/get_chats")
def get_chats():
    """Получение всех чатов пользователя"""
    db = SessionLocal()
    try:
        chats = db.query(Chat).all()

        # if not chats:
        #     raise HTTPException(status_code=404, detail="Чатов не найдено")

        return [
            {
                "user_id": chat.user_id,
                "chat_id": chat.chat_id,
                "chat": chat.chat,
                "state": chat.state
            }
            for chat in chats
        ]
    finally:
        db.close()


# Для запуска через uvicorn:
# uvicorn server:app --host 0.0.0.0 --port 5002 --reload
