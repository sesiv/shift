from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from llama_cpp import Llama
import os
import logging
from dotenv import load_dotenv
import time
import threading


logging.basicConfig(level=logging.INFO)
load_dotenv()

app = FastAPI()

# Флаг готовности модели: False, пока загрузка не завершилась
is_ready = False

# Загружаем модель в отдельном потоке, повторяя попытки до успеха.
# !pip install llama-cpp-python



def _load_model():
    global llm, is_ready
    logging.info("[question_model] loading model")
    try:
        start_time = time.time()
        llm = Llama.from_pretrained(
            repo_id="unsloth/gemma-3-4b-it-GGUF",
            filename="gemma-3-4b-it-Q4_K_M.gguf",
            set_prefix_caching=True,
            n_threads=1,
            n_ctx=1024,
            verbose=False,
            token=os.getenv("HF_TOKEN")
        )

        is_ready = True
        logging.info(f"[question_model] model loaded in {time.time() - start_time:.1f} seconds")
    except Exception as e:
        logging.exception("[question_model] failed to load model: %s", e)
        


# Запускаем загрузку в фоне при старте приложения

_load_model()
logging.info(f"[question_model] model loaded")


SYSTEM_PROMPT = """Ты ассистент Service‑Desk. Сформулируй ОДИН короткий уточняющий вопрос, 
который поможет выбрать правильную категорию работ из предложенных. 
Только вопрос, без пояснений и перечислений вариантов. Коротко и конкретно."""

@app.get("/health")
async def health_check():
    logging.info(f"[question_model] health check")
    if is_ready:
        return {"status": "ok"}
    # 503 — сервис ещё не готов
    return JSONResponse(status_code=503, content={"status": "loading"})

"""
Как должен работать /generate-question :

Вызывается на третьем уровне уверенности (70%<) 
В него передаются найденные ноды со степенью уверенности модели и вопрос пользователя 

Модель просят создать вопрос на основе их связки

На основе этого ответа модель выбирает из предложенных нод ноду пользователя и передает ее в new_main/ws/chat/{user_id}

"""



@app.post("/generate-question")
async def generate_question(data: dict):
    logging.info(f"[question_model] generate question")
    logging.info(f" received data: {data}")

    # Получаем данные из запроса
    categories = data.get("categories", [])
    user_question = data.get("question", "")
    max_tokens = min(int(data.get("max_tokens", 32)), 32)
    temperature = float(data.get("temperature", 0.7))

    # Подрежем категории и строки, чтобы уложиться в контекст
    def shorten(s: str, limit: int = 80) -> str:
        return s if len(s) <= limit else (s[:57] + " … " + s[-60:])

    trimmed = [shorten(str(c)) for c in categories[:3]]

    def build_messages(cats: list[str], question: str) -> list[dict]:
        cats_line = "; ".join([c for c in cats])
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Вопрос пользователя: {shorten(question, 160)}\nКандидаты: {cats_line}\nЗапрос: один короткий уточняющий вопрос.",
            },
        ]

    messages = build_messages(trimmed, user_question)

    try:
        output = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
    except ValueError as e:
        # На случай переполнения контекста — уменьшаем размер вывода и список нод
        if "exceed context window" in str(e):
            try:
                output = llm.create_chat_completion(
                    messages=build_messages(trimmed[:2], user_question[:120]),
                    max_tokens=12,
                    temperature=temperature,
                    stream=False,
                )
            except Exception as e2:
                logging.exception("[question_model] second attempt failed: %s", e2)
                # Fallback: simple deterministic question
                fallback_text = "Уточните, пожалуйста, с чем именно связана проблема?"
                return JSONResponse(content={
                    "choices": [{"message": {"content": fallback_text}}]
                })
        else:
            logging.exception("[question_model] generation failed: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        logging.exception("[question_model] generation failed: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

    logging.info(f" sending question: {output}")
    return JSONResponse(content=output)