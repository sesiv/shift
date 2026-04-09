# Константы конфигурации
import os

MONGO_URL = "http://mongo:5017"
E5_URL = "http://e5:5003"
VECTOR_DB_URL = "http://vector_db:5004"
QUESTION_MODEL_URL = "http://question_model:5005"
SERVER_URL = "http://server:5002"
SERVICE_PATH = "data/services.json"

# question_model.py
REPO_ID="unsloth/gemma-3-4b-it-GGUF"
MODEL_FILENAME="gemma-3-4b-it-Q4_K_M.gguf"

# e5.py
EMBEDDING_MODEL="intfloat/multilingual-e5-large-instruct"
EMBEDDING_POOLING_MODE = os.getenv("EMBEDDING_POOLING_MODE", "mean")
EMBEDDING_IDF_PATH = os.getenv("EMBEDDING_IDF_PATH", "")
EMBEDDING_POOLING_CHECKPOINT = os.getenv("EMBEDDING_POOLING_CHECKPOINT", "")
EMBEDDING_POOLING_ALPHA = float(os.getenv("EMBEDDING_POOLING_ALPHA", "1.0"))
EMBEDDING_QUANTIZE = os.getenv("EMBEDDING_QUANTIZE", "1").lower() not in {"0", "false", "no"}

# Верхний и нижний пороги уверенности модели
CONFIDENCE_CONSTANTS = [0.83, 0.5]

# Калибровка (нужна для определения уверенности модели)
GLOBAL_CALIBRATION = [
            (0.0019, 0.9891),
            (0.0449, 0.9710),
            (0.0577, 0.9559),
            (0.0655, 0.9420),
            (0.0726, 0.8986),
            (0.1078, 0.8261),
            (0.1209, 0.8116),
            (0.1331, 0.8088),
            (0.1442, 0.7101),
            (0.1554, 0.6667),
            (0.1832, 0.5507),
            (0.2003, 0.4493),
            (0.2844, 0.3478)
        ]

# Минимальная разница между score при выборе id с максимальным score
SCORE_DELTA = 0.01

# Если число уточняющих вопросов >= CLARIFICATION_COUNT_TRESHOLD,
# то обработать вопрос юзера функцией _handle_medium_confidence_response
CLARIFICATION_COUNT_TRESHOLD = 1

#Креды для подключения к chroma
CHROMA_CLIENT_AUTH_PROVIDER="chromadb.auth.basic_authn.BasicAuthClientProvider"
CHROMA_SERVER_PORT=8000
CHROMA_SERVER_HOST="158.160.17.124"
CHROMA_COLLECTION_NAME="ticketsTrain09082025"

# Количество возвращаемых результатов из векторной бд
N_RESULTS = 5

# Максимальная длина вектора при генерации для токенайзера e5
MAX_VECTOR_LENGTH = 512


