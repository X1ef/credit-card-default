import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
import time

from flask import Flask, request, jsonify
from model_handler import load_model, predict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

logger.info("Загрузка модели...")
MODEL = load_model()
logger.info("Модель загружена.")

START_TIME = time.time()


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':         'healthy',
        'uptime_seconds': round(time.time() - START_TIME, 1),
    }), 200


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Тело запроса должно быть валидным JSON'}), 400

    logger.info({'event': 'predict_request', 'input_keys': list(data.keys())})

    try:
        result = predict(MODEL, data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    interpretation = (
        'Клиент совершит дефолт в следующем месяце'
        if result['prediction'] == 1
        else 'Клиент не совершит дефолт в следующем месяце'
    )

    response = {
        'prediction':     result['prediction'],
        'probability':    result['probability'],
        'interpretation': interpretation,
    }

    logger.info({'event': 'predict_response', **result})
    return jsonify(response), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
