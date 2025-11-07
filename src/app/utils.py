"""
Вспомогательные функции для приложения Service Desk.
"""

import bisect
from consts import (
    GLOBAL_CALIBRATION,
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
