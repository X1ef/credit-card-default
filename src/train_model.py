import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TARGET_COL = 'default.payment.next.month'
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')


def load_data(filepath: str) -> pd.DataFrame:
    logger.info(f"Загрузка данных из {filepath}")

    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        # UCI-датасет иногда приходит как Excel с лишней строкой заголовка
        df = pd.read_excel(filepath, header=1)
    else:
        df = pd.read_csv(filepath)

    # Удаляем колонку ID если есть
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    logger.info(f"Датасет загружен: {df.shape}, дефолт rate: {df[TARGET_COL].mean():.3f}")
    return df


def train(df: pd.DataFrame) -> None:
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Обучение GradientBoostingClassifier...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    logger.info(f"F1-macro на тесте: {f1:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        'model':    model,
        'features': list(X.columns),
        'f1_macro': round(f1, 4),
    }, MODEL_PATH)
    logger.info(f"Модель сохранена: {MODEL_PATH}")


def main():
    csv_path = '/Users/mac1/Downloads/project/data/UCI_Credit_Card.csv'
    df = load_data(csv_path)
    train(df)


if __name__ == '__main__':
    main()
