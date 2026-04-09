# Отчет по внедрению TF-IDF pooling для `multilingual-e5-large-instruct`

## Что реализовано

В задаче классификации и поиска похожих обращений ключевую роль играет sentence embedding, то есть компактный числовой вектор, который должен сохранять смысл всего обращения и позволять сравнивать тексты между собой в общем метрическом пространстве. Если такой вектор построен удачно, то обращения одной и той же категории располагаются ближе друг к другу, а обращения разных категорий оказываются дальше. В исходной постановке итоговый вектор получался обычным усреднением скрытых состояний всех токенов, поэтому вклад информативных и малоинформативных фрагментов текста был примерно одинаковым. Идея модификации состояла в том, чтобы сделать агрегирование содержательно неоднородным и усилить влияние тех токенов, которые действительно различают классы заявок. Для этого в библиотеке `transformers` была введена TF-IDF-взвешенная схема pooling, в которой итоговый sentence embedding вычисляется как взвешенное среднее `v = sum(w_i * h_i) / sum(w_i)`, где `w_i = 1 + alpha * tfidf_i`. Величина TF-IDF в этой формуле отражает, насколько токен важен одновременно для конкретного текста и для всего корпуса, а параметр `alpha` регулирует силу этого эффекта. За счет такого подхода редкие и семантически значимые токены получают больший вес при построении эмбеддинга, чем часто встречающиеся и менее различающие элементы текста. Чтобы эксперимент оставался корректным, IDF вычислялся только по train-корпусу, а обучение было сформулировано как ranking-задача, в которой запрос должен быть ближе к positive-примеру своего класса и дальше от negative-примера другого класса.

По текущим сохраненным результатам baseline mean pooling на validation дает `top1 = 0.8104`, `top3 = 0.8913`, `MRR = 0.8579`, а на test `top1 = 0.8168`, `top3 = 0.9106`, `MRR = 0.8686`. Для TF-IDF pooling без дообучения в текущем экспериментальном файле зафиксированы более высокие значения `top1` и `top3`: на validation `top1 = 0.8226`, `top3 = 0.9211`, `MRR = 0.8890`, а на test `top1 = 0.8363`, `top3 = 0.9293`. Таким образом, по двум основным retrieval-метрикам наблюдается улучшение относительно baseline как на validation, так и на test, что подтверждает практическую полезность перераспределения весов токенов при построении sentence embedding. На validation улучшение прослеживается и по `MRR`, что дополнительно указывает на рост качества ранжирования релевантных обращений. При этом значение `MRR = 0.0292`, записанное для test в текущем файле результатов, не согласуется с одновременно высокими `top1` и `top3`, поэтому данная конкретная величина требует отдельной перепроверки перед включением в финальную таблицу диплома. В остальном текущий набор метрик позволяет сделать предварительный вывод о том, что TF-IDF-взвешенный pooling способен улучшать качество поиска близких обращений уже на этапе построения эмбеддингов.

### 1. Изменен исходный модуль формирования sentence embedding
- Добавлен локальный fork model-класса `XLMRobertaModel` в `src/app/modeling_xlm_roberta.py`.
- Теперь sentence embedding формируется внутри этого локального model-class, а не во внешней project-side обертке.
- В форке стандартный `mean pooling` сохранен как режим `mean`.
- Добавлен новый режим `tfidf_weightedmean`.
- Реализована формула:
  - `w_i = 1 + alpha * tfidf_i`
  - `v = sum(w_i * h_i) / sum(w_i)`
- `alpha` сделан обучаемым параметром pooling-модуля.
- `attention_mask` учитывается и в сумме эмбеддингов, и в сумме весов.
- PAD-токены не участвуют в агрегации.
- L2-нормализация после pooling сохранена без изменений.
- Добавлена загрузка `idf`-артефакта и checkpoint pooling/последнего transformer block.
- В `src/app/e5.py` заменен импорт модели: вместо голого `AutoModel/XLMRobertaModel` используется локальный forked class.

### 2. Добавлен подготовительный контур для датасета
- В `src/app/xlsx_reader.py` добавлен stdlib-based reader для `.xlsx`, без `openpyxl`.
- В `src/app/e5_experiment_data.py` добавлены:
  - загрузка тикетов из `ExportSDLab.xlsx`;
  - очистка и сборка текста из `Тема + Описание`;
  - разбиение на `train/validation/test`;
  - генерация `positive` и `negative` pairs;
  - генерация triplets для ranking loss;
  - расчет `idf` только по train-корпусу;
  - сериализация артефактов в `json/jsonl`.
- Уровень весов выбран `token_id`, чтобы pooling использовал те же токены, что и модель.

### 3. Добавлен training/evaluation pipeline
- `src/app/e5_prepare_data.py`:
  - готовит split'ы;
  - сохраняет `*_records.jsonl`, `*_pairs.jsonl`, `*_triplets.jsonl`;
  - сохраняет `idf_token_id.json`.
- `src/app/e5_train.py`:
  - замораживает весь encoder;
  - обучает только pooling-модуль;
  - использует `TripletMarginLoss`;
  - опционально умеет разморозить только последний transformer block.
- `src/app/e5_evaluate.py`:
  - сравнивает:
    - baseline `mean pooling`,
    - `TF-IDF pooling` без дообучения,
    - `TF-IDF pooling` после дообучения;
  - считает retrieval-метрики `top1`, `top3`, `MRR`.

### 4. Добавлен editable install и документация
- Добавлен `pyproject.toml` для `pip install -e .`.
- Добавлен `src/config/requirements/embedding_experiments.txt`.
- `README.md` обновлен командами для:
  - editable install;
  - подготовки данных;
  - обучения;
  - сравнения режимов;
  - переключения pooling через env vars.

## Что проверено локально

### Статические и unit-проверки
- `python -m compileall src/app src/tests/unit` — успешно.
- `python -m unittest src.tests.unit.test_e5_pipeline` — успешно, 6 тестов.

### Smoke-check на реальных данных `ExportSDLab.xlsx`
- Загружено записей: `11540`.
- После split:
  - `train`: `9249` записей, `207` классов;
  - `validation`: `1150` записей, `137` классов;
  - `test`: `1141` записей, `128` классов.
- Генерация артефактов на реальных данных отрабатывает:
  - `train`: `32988` pairs, `9223` triplets;
  - `validation`: `8720` pairs, `1098` triplets;
  - `test`: `8720` pairs, `1098` triplets.

## Ограничения текущей среды

- Полный запуск `e5_prepare_data.py / e5_train.py / e5_evaluate.py` не выполнялся до конца, потому что для tokenizer/model `intfloat/multilingual-e5-large-instruct` нужен доступ к Hugging Face или заранее прогретый локальный cache.
- Проверка `pip install -e .` в этой среде частично подтверждена:
  - editable wheel успешно собирается через `pip install -e . --no-build-isolation`;
  - финальная установка блокируется правами на текущий `site-packages` виртуального окружения (`Permission denied`), то есть это ограничение окружения, а не структуры проекта.

## Основные файлы

- `src/app/modeling_xlm_roberta.py`
- `src/app/e5.py`
- `src/app/e5_experiment_data.py`
- `src/app/e5_prepare_data.py`
- `src/app/e5_train.py`
- `src/app/e5_evaluate.py`
- `src/app/xlsx_reader.py`
- `src/app/consts.py`
- `src/config/requirements/embedding_experiments.txt`
- `pyproject.toml`
- `README.md`

## Рекомендуемая последовательность запуска

```bash
pip install -r src/config/requirements/embedding_experiments.txt
pip install -e . --no-build-isolation

python src/app/e5_prepare_data.py \
  --source src/data/ExportSDLab.xlsx \
  --output-dir data/e5_pooling

python src/app/e5_train.py \
  --dataset-dir data/e5_pooling \
  --output-dir data/e5_pooling/checkpoints

python src/app/e5_evaluate.py \
  --dataset-dir data/e5_pooling \
  --checkpoint-path data/e5_pooling/checkpoints/best_pooling_checkpoint.pt
```
