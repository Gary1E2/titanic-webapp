# training_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from training import train_model

app = FastAPI()

class TrainReq(BaseModel):
    train_clean_filename: str = "train_clean.csv"
    model_filename: str = "model.pkl"

@app.post("/train")
def train(req: TrainReq):
    metrics = train_model(req.train_clean_filename, req.model_filename)
    return {"status": "ok", **metrics}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
