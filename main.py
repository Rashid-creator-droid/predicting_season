import io

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from ml.loaders import DatasetManager
from ml.predict import SeasonPredictor

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

predictor = SeasonPredictor()
dataset_manager = DatasetManager()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    season, probability = predictor.predict(image)
    return {"season": season, "probability": probability}


@app.get("/test_dataset")
async def test_dataset():
    cm_image_base64, accuracy = predictor.test_model()
    return {"cm_image": cm_image_base64, "accuracy": accuracy}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, port=8001)
