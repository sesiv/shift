"""
Вспомогательные функции для приложения Service Desk.

Этот модуль содержит функции-помощники для векторных операций,
управления узлами, калибровки и взаимодействия с внешними сервисами.
"""

import json
import logging
import requests
import bisect
from typing import List
from consts import (
    GLOBAL_CALIBRATION,
    VECTOR_DB_URL,
    E5_URL,
    SCORE_DELTA
)


def distance_to_confidence(distance_value: float) -> float:
    """
    Преобразовать значение расстояния в оценку уверенности используя данные калибровки.

    Args:
        distance_value: Значение расстояния для преобразования

    Returns:
        Оценка уверенности от 0 до 1
    """
    if not GLOBAL_CALIBRATION:
        return 0.0
    distances = [d for d, _ in GLOBAL_CALIBRATION]
    idx = bisect.bisect_left(distances, distance_value)
    if idx <= 0:
        return GLOBAL_CALIBRATION[0][1]
    if idx >= len(GLOBAL_CALIBRATION):
        return GLOBAL_CALIBRATION[-1][1]
    d1, p1 = GLOBAL_CALIBRATION[idx - 1]
    d2, p2 = GLOBAL_CALIBRATION[idx]
    if d2 == d1:
        return p1
    # Линейная интерполяция между точками
    alpha = (distance_value - d1) / (d2 - d1)
    return p1 + alpha * (p2 - p1)


def aggregate_nodes(state: str, message: str) -> dict:
    """
    Классифицирует сообщение в один из узлов и вычисляет откалиброванную уверенность.

    Args:
        state: Текущее состояние разговора
        message: Сообщение пользователя для классификации

    Returns:
        Словарь с ключами:
        - predicted_id: str | None — предсказанный идентификатор узла
        - confidence: float в диапазоне [0,1] — уверенность в предсказании
        - top_categories: список словарей {id, score}, отсортированных по убыванию score
        - best_distance: float | None — минимальная дистанция для предсказанной категории
    """
    message_vector = get_vector(message)
    similar_nodes = search_similar_nodes(state, message_vector)
    similar_nodes_dict = json.loads(similar_nodes.get("response", "[]"))
    logging.info(f"similar_nodes_dict {similar_nodes_dict}")

    hits: dict = {"folder": {}, "slmService": {}, "categoriesWork": {}}

    if state == "baseState":
        for node in similar_nodes_dict:
            hits["folder"][node["folder"]] = (
                hits["folder"].get(node["folder"], 0) + node["distance"]
            )
            hits["slmService"][node["slmService"]] = (
                hits["slmService"].get(node["slmService"], 0) + node["distance"]
            )
            hits["categoriesWork"][node["categoriesWork"]] = (
                hits["categoriesWork"].get(node["categoriesWork"], 0) + node["distance"]
            )
    elif state == "folder":
        for node in similar_nodes_dict:
            hits["slmService"][node["slmService"]] = (
                hits["slmService"].get(node["slmService"], 0) + node["distance"]
            )
            hits["categoriesWork"][node["categoriesWork"]] = (
                hits["categoriesWork"].get(node["categoriesWork"], 0) + node["distance"]
            )
    elif state == "slmService":
        for node in similar_nodes_dict:
            hits["categoriesWork"][node["categoriesWork"]] = (
                hits["categoriesWork"].get(node["categoriesWork"], 0) + node["distance"]
            )

    logging.info(f"hits {hits}")

    best = {
        "folder": {"id": "", "score": 0.0},
        "slmService": {"id": "", "score": 0.0},
        "categoriesWork": {"id": "", "score": 0.0},
    }

    for level in hits:
        for hit in hits[level]:
            if best[level]["score"] < hits[level][hit]:
                best[level]["score"] = hits[level][hit]
                best[level]["id"] = hit

    max_score = max(best[level]["score"] for level in best)
    logging.info(f"max score {max_score}")
    logging.info(f"best {best}")

    predicted_id = None
    # Приоритетно выбираем id с максимальным score среди уровней: категории работ, услуга, папка
    for priority in ["categoriesWork", "slmService", "folder"]:
        if abs(best[priority]["score"] - max_score) < SCORE_DELTA:
            predicted_id = best[priority]["id"]
            break

    # Определяем минимальную дистанцию для предсказанной категории для калибровки уверенности
    best_distance = None
    if predicted_id:
        candidate_distances: List[float] = []
        # Собираем дистанции для нод, у которых совпадает predicted_id на любом уровне
        for node in similar_nodes_dict:
            if (
                node.get("categoriesWork") == predicted_id
                or node.get("slmService") == predicted_id
                or node.get("folder") == predicted_id
            ):
                try:
                    candidate_distances.append(float(node["distance"]))
                except Exception:
                    pass
        if candidate_distances:
            best_distance = min(candidate_distances)
        else:
            # Если не найдено — используем минимальную дистанцию среди всех нод
            try:
                best_distance = min(float(n["distance"]) for n in similar_nodes_dict)
            except Exception:
                best_distance = None

    confidence = (
        distance_to_confidence(best_distance) if best_distance is not None else 0.0
    )

    # Формируем топ категорий (по убыванию агрегированного score) для подсказок
    categories_scores = hits["categoriesWork"]
    sorted_categories = sorted(
        categories_scores.items(), key=lambda x: x[1], reverse=True
    )
    top_categories = [{"id": cid, "score": score} for cid, score in sorted_categories]

    result = {
        "predicted_id": predicted_id,
        "confidence": confidence,
        "top_categories": top_categories,
        "best_distance": best_distance,
    }
    return result


def get_vector(text: str):
    """
    Получает векторное представление текста используя модель E5.

    Args:
        text: Входной текст для векторизации

    Returns:
        Векторное представление текста
    """
    response = requests.post(f"{E5_URL}/get_vector", json={"query": text})
    return response.json()["vector"]


def search_similar_nodes(state, vector):
    """
    Выполняет поиск похожих узлов используя векторное сходство.

    Args:
        state: Текущее состояние
        vector: Вектор запроса

    Returns:
        Ответ от сервиса векторной базы данных
    """
    response = requests.post(
        f"{VECTOR_DB_URL}/ticket/search", json={"state": state, "query_vector": vector}
    )
    return response.json()
