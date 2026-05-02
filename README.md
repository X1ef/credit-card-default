# Credit Card Default Prediction Service

Сервис машинного обучения для предсказания дефолта по кредитным картам.
Реализует полный production-пайплайн: обучение модели → Flask API → Docker.

**Датасет:** [Default of Credit Card Clients (UCI / Kaggle)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  
**Модель:** GradientBoostingClassifier (scikit-learn)  
**Домен:** финансы / кредитный скоринг

---

## Структура репозитория

```
credit-card-default/
├── src/
│   ├── api.py              # Flask-приложение
│   ├── model_handler.py    # Загрузка и инференс модели
│   └── train_model.py      # Скрипт обучения
├── models/
│   └── model.pkl           # Обученная модель (joblib)
├── tests/
│   ├── test_api.py         # Pytest-тесты
│   └── *.png               # Скриншоты (PNG) с демонстрацией работы  
├── data/
│   └── UCI_Credit_Card.csv # Датасет
├── Dockerfile
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

---

## Быстрый старт

### 1. Клонировать и перейти в папку

```bash
git clone https://github.com/x1ef/credit-card-default.git
cd credit-card-default
```

### 2. Создать окружение и установить зависимости

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Импорт датасета с локали


### 4. Обучить модель

```bash
python src/train_model.py
```
Обученная модель сохранится в `models/model.pkl`.

### 5. Запустить сервис

```bash
PORT=8080 python src/api.py
```

Сервис запустится на `http://localhost:8080`.

---

## API

### GET /health

```bash
curl http://localhost:8080/health
```

```json
{"status": "healthy", "uptime_seconds": 31.7}
```

### POST /predict

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000,
    "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000,
    "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
    "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000
  }'
```

**Ответ:**
```json
{
  "prediction": 0,
  "probability": 0.1303,
  "interpretation": "The client will not default next month"
}
```


**Коды ответов:** `200` — успех, `400` — ошибка входных данных.

---

## Docker

### Сборка и запуск

```bash
docker build -t credit-default-api .
docker run -p 5001:5000 credit-default-api
```

### Docker Hub

```bash
docker tag credit-default-api x1ef/credit-default-api:latest
docker push x1ef/credit-default-api:latest
```

Образ: `docker pull x1ef/credit-default-api:latest`

Ссылка на docker-образ: https://hub.docker.com/r/x1ef/credit-default-api

---

## Тесты

```bash
pip install pytest
pytest tests/test_api.py -v
```

---

## A/B-тестирование

### Постановка теста

Предположим, обучена улучшенная модель v2 (например, с другими гиперпараметрами или дополнительными признаками). Тест сравнивает:

| | Контрольная группа (A) | Тестовая группа (B) |
|---|---|---|
| Модель | текущая (v1) | новая (v2) |
| Доля трафика | 50% | 50% |
| Длительность | 4 недели | 4 недели |

Разделение клиентов по группам — случайное, по хешу `client_id`:
```python
group = "B" if hash(client_id) % 2 == 0 else "A"
```

### Метрики оценки

**Основная — F1-score для класса дефолта (class = 1).**  
Выбор обоснован: классы несбалансированы (доля дефолтов около 22%), поэтому accuracy не подходит. F1 учитывает и Precision, и Recall одновременно.

**Дополнительная — Recall для класса дефолта.**  
Пропущенный дефолт (ложноотрицательный результат) обходится банку дороже, чем ложный отказ в кредите. Поэтому Recall важнее Precision.

**Бизнес-метрики:**
1. Ожидаемые финансовые потери: `Σ (probability_i × balance_i)` — чем точнее модель калибрует вероятности, тем точнее банк резервирует капитал.
2. Доля одобренных заявок при пороге риска ≤ 0.3 — контролирует, не стала ли новая модель чрезмерно консервативной.

### Статистический анализ

- Для F1-score — **bootstrap** (10 000 итераций), так как у F1 нет аналитического распределения.
- Для Recall — **z-тест для двух пропорций** (H₀: метрики равны, α = 0.05).
- Доверительные интервалы: `p ± 1.96 × √(p(1−p)/n)`.

### Критерий успешности

Переход на v2, если выполнены все три условия:
1. F1 у v2 выше на ≥ 0.02 (условно).
2. p-value z-теста для Recall < 0.05.
3. Доля одобренных заявок у v2 не снизилась более чем на 2% (аналогично п.1, условно).

### Архитектурная реализация

В текущем сервисе роутинг реализуется на уровне API Gateway: по `client_id` запрос направляется к нужной версии модели. Логи каждого запроса содержат версию модели и результат предсказания — после окончания теста они агрегируются для расчёта метрик по группам.
