import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from api import app

SAMPLE = {
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000,
    "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000,
    "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
    "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000,
}


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'healthy'


def test_predict_ok(client):
    resp = client.post('/predict', json=SAMPLE)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['prediction'] in [0, 1]
    assert 0.0 <= data['probability'] <= 1.0
    assert 'interpretation' in data


def test_predict_missing_feature(client):
    resp = client.post('/predict', json={"LIMIT_BAL": 50000})
    assert resp.status_code == 400
    assert 'error' in resp.get_json()


def test_predict_invalid_json(client):
    resp = client.post('/predict', data='not json', content_type='application/json')
    assert resp.status_code == 400
