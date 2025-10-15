## Установка 

1) склонировать репо - > 
``` bash
cd src/config
docker compose up 
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

