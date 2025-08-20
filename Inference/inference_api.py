# inference_api.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from inference import predict_from_clean

app = FastAPI()

class InferReq(BaseModel):
    test_clean_filename: str = "test_clean.csv"
    model_filename: str = "model.pkl"
    output_filename: str = "test_with_preds.csv"
    target_col: str = "Survived"

@app.post("/predict")
def predict(req: InferReq):
    out = predict_from_clean(
        req.test_clean_filename, req.model_filename, req.output_filename, req.target_col
    )
    return {"status": "ok", "predictions_file": out}

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
