## Установка 

1) склонировать репо - > 
``` bash
cd src/config
docker compose up 
```

## Структура проекта 

```text
Service_Desk/
├── docs/                 содержит описание работы бота
├── src/
│   ├── notebooks/        папка для ноутбуков
│   ├── app/              папка для кода
│   ├── config/           папка для конфигов
│   ├── data/             папка для датасетов, json
│   └── tests/            папка для тестов
├── README.md             описание
├── CHANGELOG.md          версионирование проекта
└── requirements.txt      содержит описание либ проекта
```

## Запуск 

``` bash 
cd src/config
docker compose up
``` 

Нужно установить в .env
- свой HF_TOKEN в https://huggingface.co/settings/tokens
- MONGO_URI - креды для подключения к mongo серверу 
- CHROMA_CLIENT_AUTH_CREDENTIALS - креды для подключения к chroma серверу


ВАЖНО при первом запуске надо поставить параметр question_model/healthcheck/
timeout: 10s на больше ,  минут 10 или 15 для загрузки модели локально 

или альтернативно запустить вне контейнера и подождать пока загрузится 

``` bash
uvicorn question_model:app --reload 
``` 

при проблемах с контейнерами зачастую помогает

``` bash 
docker compose up --remove-orphans
``` 

## Документация 
Swagger генерирует документацию для открытых http эндпоинтов

``` bash
http://localhost:8000/docs (главный API),
http://localhost:5002/docs (server), 
http://localhost:5004/docs (vector_db), 
http://localhost:5017/docs (mongo), 
http://localhost:5005/docs (question_model).
``` 
