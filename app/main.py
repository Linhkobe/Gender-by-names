"""
main.py - FastAPI application entry point for gender prediction based on first name.

this file initiates and runs the FastAPI web application. It provides : 

1. REST API endpoints
- GET /predict -> takes a first name and returns the predicted gender and probabilities.

2. Web interface (FRONTEND)
- GET / -> serves the main HTML page with a form to input a first name and display results.

3. Model loading 
- Loads a pre-trained tensorFlow model in 'model/gender_model.keras'

The app uses Jinja2 templates for rendering HTML and serves static files from the 'static' directory.

Run locally with: 
    uvicorn app.main:app --reload
    
In production (Docker, Kubernetes), use: 
uvicorn app.main:app --host 0.0.0.0 --port 8080
"""

import os
import tensorflow as tf
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydantic import Field, BaseModel, model_validator

# ===================
# Initiate FastAPI app
# ===================
app = FastAPI(title = "Gender Prediction API", version = "1.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

# ===================
# Load the pre-trained model
# ===================
MODEL_PATH = "model/prod/exported_model"
# model = tf.keras.models.load_model(MODEL_PATH)

# ===================
# Request schema
# ===================
class NameRequest(BaseModel):
    preusuel: str | None = Field(default= None)
    name: str | None = Field(default= None)
    
    def _pick_field(self):
        v = (self.preusuel or self.name or "").strip()
        if not v:
            raise ValueError("Either 'preusuel' or 'name' must be provided and non-empty.")
        self.preusuel = v
        return self

# ===================
# API Endpoints
# ===================
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from app.service import service
@app.post("/predict")
async def predict(req: NameRequest):
    return service.predict_label(req.preusuel)


@app.get("/health")
async def health():
    return {"status": "ok"}

from app.service import service, name_encoder
@app.get("/ready")
async def ready():
    try:
        model_ok = hasattr(service, "model") and service.model is not None
        enc_ok = hasattr(name_encoder, "_map") and isinstance(name_encoder._map, dict) and len(name_encoder._map) > 0
        prior_ok = hasattr(name_encoder, "_prior")
        
        if model_ok and enc_ok and prior_ok:
            return {"status": "ready", "model": True, "name_encoder": True}
        else:
            return JSONResponse(status_code=503, content={"status": "not ready", "model": model_ok, "name_encoder": enc_ok and prior_ok})
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "not ready", "error": str(e)})