from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from pydantic import BaseModel
from pathlib import Path


MODEL_PATH = Path("model/nlp_model")
FRONTEND_INDEX = Path("static/index.html")

print("========== MODEL_PATH ", MODEL_PATH)
print("========== FRONTEND_INDEX ", FRONTEND_INDEX)

app = FastAPI(title="Spam Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_headers = ["*"],
    allow_methods = ["*"],
)

app.mount("/static", StaticFiles(directory='static'), name='static')
# app.mount("/static", StaticFiles(directory='static'), name="static")


class Message(BaseModel):
    message: str

model_data = {}

@app.on_event('startup')
def load_model():
    global model_data

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=500, detail=f"Model file {MODEL_PATH} not found. Run model.py first.")

    
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
   
@app.get('/')
def index():
    return FileResponse(FRONTEND_INDEX)


@app.post('/predict')
def predict(payload: Message):

    message = payload.message

    try:        
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        target_names = model_data['target_names']

        message_v = vectorizer.transform([message])

        # model expect 2D array
        pred = model.predict(message_v)[0]
        proba = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(message_v)[0].tolist()
        else:
            proba=None

        return {
                "prediction_index": int(pred),
                "prediction": str(target_names[int(pred)]),
                "probabilities": proba,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))