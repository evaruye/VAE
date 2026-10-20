"""Run FastAPI endpoint smoke tests without starting a browser or public server.

Usage (from repository root): python app/smoke_test.py
Uses FastAPI TestClient, which requires httpx (listed in app/requirements.txt).
"""
from __future__ import annotations
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from fastapi.testclient import TestClient
from backend.main import app


def main():
    with TestClient(app) as client:
        assert client.get('/').status_code == 200, 'Frontend cannot be served'
        health = client.get('/api/health')
        assert health.status_code == 200
        h = health.json()
        assert h.get('model_ready') is True, h.get('model_init_error')
        assert h.get('raw_formula_ready') is True, 'Raw feature builder failed'
        schema = client.get('/api/raw-schema')
        assert schema.status_code == 200 and schema.json()['ready'] is True
        example = client.get('/api/raw-example')
        assert example.status_code == 200
        raw_prediction = client.post('/api/raw-predict', json={'raw_formula':example.json()['raw_formula']})
        assert raw_prediction.status_code == 200, raw_prediction.text
        p = raw_prediction.json()['predictions']
        expected = {
            'viscosity': 328.8125469720,
            'solid_content': 49.9101104736,
            'ethylene_content': 22.4476574110,
            'free_hcho': 23.0649377033,
        }
        for target, ref in expected.items():
            got = float(p[target]['value'])
            assert abs(got - ref) < max(0.05, abs(ref) * 0.001), f'{target}: {got} vs {ref}'
        features_example = client.get('/api/example').json()['features']
        feature_prediction = client.post('/api/predict', json={'features':features_example})
        assert feature_prediction.status_code == 200, feature_prediction.text
        bad_request = client.post('/api/raw-predict',json={'raw_formula': {}})
        assert bad_request.status_code == 400, bad_request.text
        print('PASS: frontend, model health, raw schema, raw example, raw prediction, feature prediction, invalid-input check')
        for k in ('viscosity','solid_content','ethylene_content','free_hcho'):
            print(f'  {k}: {p[k]["value"]:.6f} {p[k]["unit"]}')
        if h['models'].get('load_warnings'):
            print('NOTE: version compatibility warnings were emitted; use pinned package versions for publication reproducibility')


if __name__ == '__main__':
    main()
