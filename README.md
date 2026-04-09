## Установка 

1) склонировать репо - > 
``` bash
cd src/config
docker compose up 
```

Для локальной разработки и правки pooling-модуля без переустановки добавлен editable install:
```bash
pip install -e .
```

Если окружение офлайн или `pip` не может скачать build dependencies, используйте:
```bash
pip install -e . --no-build-isolation
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


## Запуск тестов 
Для запуска тестов нужно виртуальное окружение с pytest
``` bash 
pip install -r src/config/requirements/tests.txt
```

``` bash
cd src/config
docker compose up
```
Дождитесь уведомления от докера что все загрузилось 

``` bash
pytest src/tests/integration/main.py -s
```

## TF-IDF pooling для E5

Для экспериментального контура добавлены отдельные скрипты подготовки датасета, обучения и оценки.
Sentence embedding теперь формируется не внешней оберткой, а локальным forked model-class в `src/app/modeling_xlm_roberta.py`, который заменяет project-side вызов голого `XLMRobertaModel`.

Установка зависимостей:
```bash
pip install -r src/config/requirements/embedding_experiments.txt
pip install -e .
```

Подготовка `train/validation/test`, positive/negative пар, triplets и `idf` только по train:
```bash
python src/app/e5_prepare_data.py \
  --source src/data/ExportSDLab.xlsx \
  --output-dir data/e5_pooling
```

Обучение нового pooling-модуля при замороженном encoder:
```bash
python src/app/e5_train.py \
  --dataset-dir data/e5_pooling \
  --output-dir data/e5_pooling/checkpoints
```

Сравнение трех режимов:
```bash
python src/app/e5_evaluate.py \
  --dataset-dir data/e5_pooling \
  --checkpoint-path data/e5_pooling/checkpoints/best_pooling_checkpoint.pt
```

Во время рантайма векторного сервиса можно переключать режимы через переменные окружения:
- `EMBEDDING_POOLING_MODE=mean`
- `EMBEDDING_POOLING_MODE=tfidf_weightedmean`
- `EMBEDDING_IDF_PATH=/path/to/idf_token_id.json`
- `EMBEDDING_POOLING_CHECKPOINT=/path/to/best_pooling_checkpoint.pt`
- `EMBEDDING_POOLING_ALPHA=1.0`



## Документация 
Swagger генерирует документацию для открытых http эндпоинтов

``` bash
http://localhost:8000/docs (главный API),
http://localhost:5002/docs (server), 
http://localhost:5004/docs (vector_db), 
http://localhost:5017/docs (mongo), 
http://localhost:5005/docs (question_model).
``` 
