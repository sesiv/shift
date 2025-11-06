FROM python:3.12-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

EXPOSE 8501 5004 5017

FROM base AS streamlit
COPY src/config/requirements/web.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r web.txt
COPY . .
CMD ["streamlit", "run", "web.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

FROM base AS main
COPY src/config/requirements/main.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r main.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS server
COPY src/config/requirements/server.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r server.txt
COPY . .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5002"]

# Build стадия для компиляции ML зависимостей
FROM base AS question_model-builder
RUN apt-get update && apt-get install -y \
    gcc g++ cmake make git pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY src/config/requirements/question_model.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r question_model.txt
# Финальная стадия для question_model (только рантайм)
FROM base AS question_model
RUN apt-get update && apt-get install -y \
    libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=question_model-builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=question_model-builder /usr/local/bin/ /usr/local/bin/
COPY . .
RUN python -c "import llama_cpp; print('ML dependencies verified')"
CMD ["uvicorn", "question_model:app", "--host", "0.0.0.0", "--port", "5005"]

FROM base AS mongo
COPY src/config/requirements/mongo.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r mongo.txt
COPY . .
CMD ["uvicorn", "mongo:app", "--host", "0.0.0.0", "--port", "5017"]

FROM base AS vector_db
COPY src/config/requirements/vector_db.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r vector_db.txt
COPY . .
CMD ["uvicorn", "vector_db:app", "--host", "0.0.0.0", "--port", "5004"]