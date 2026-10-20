from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .model_service import VAEModelService
from .raw_formula_service import RawFormulaService

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="VAE Emulsion Property Prediction API", version="3.7.0")

service = None
model_init_error = None
try:
    service = VAEModelService()
except Exception as exc:
    model_init_error = str(exc)

raw_service = RawFormulaService(service)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class RawPredictionRequest(BaseModel):
    raw_formula: Dict[str, Any]


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": service is not None,
        "model_init_error": model_init_error,
        "raw_formula_ready": raw_service.ready,
        "models": service.model_info() if service is not None else {},
    }


@app.get("/api/schema")
def schema():
    if service is None:
        return {"config": {}, "models": {}, "model_init_error": model_init_error}
    return {"config": service.config, "models": service.model_info()}


@app.get("/api/raw-schema")
def raw_schema():
    return raw_service.schema()


@app.get("/api/example")
def example():
    if service is None:
        raise HTTPException(status_code=503, detail="Model files are not available in this package.")
    return {"features": service.config["example"]}


@app.get("/api/raw-example")
def raw_example():
    try:
        return raw_service.example()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/predict")
def predict(req: PredictionRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model files are not available in this package.")
    try:
        return service.predict(req.features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}")


@app.post("/api/raw-predict")
def raw_predict(req: RawPredictionRequest):
    try:
        return raw_service.predict(req.raw_formula)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Raw-formulation prediction failed: {exc}")
