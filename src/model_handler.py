"""
Загрузка модели и инференс.
"""

import os
import logging
import numpy as np
import joblib
from typing import Dict, Any

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')


def load_model() -> Dict[str, Any]:
    """Загружает модель из models/model.pkl."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}\n"
            "Сначала запустите: python src/train_model.py"
        )
    artifact = joblib.load(MODEL_PATH)
    logger.info(f"Модель загружена. F1-macro при обучении: {artifact.get('f1_macro', 'N/A')}")
    return artifact


def predict(artifact: Dict[str, Any], input_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Выполняет предсказание.

    Args:
        artifact: словарь с ключами 'model', 'features'
        input_data: словарь признаков клиента

    Returns:
        dict с 'prediction' (0/1) и 'probability' (float)
    """
    model    = artifact['model']
    features = artifact['features']

    try:
        feature_vector = np.array([input_data[f] for f in features]).reshape(1, -1)
    except KeyError as e:
        raise ValueError(f"Отсутствует обязательный признак: {e}")

    prediction  = int(model.predict(feature_vector)[0])
    probability = float(model.predict_proba(feature_vector)[0][1])

    return {
        'prediction':  prediction,
        'probability': round(probability, 4),
    }
