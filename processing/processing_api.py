import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from processing import preprocess_pair, preprocess_single

app = FastAPI()

class PairReq(BaseModel):
    train_in: str = "train.csv"
    test_in: str  = "test.csv"
    train_out: str = "train_clean.csv"
    test_out: str  = "test_clean.csv"

class SingleReq(BaseModel):
    input_filename: str
    output_filename: str = "cleaned.csv"

@app.post("/process/pair")
def process_pair(req: PairReq):
    tr_out, te_out = preprocess_pair(req.train_in, req.test_in, req.train_out, req.test_out)
    return {"status": "ok", "train_clean": tr_out, "test_clean": te_out}

@app.post("/process/single")
def process_single(req: SingleReq):
    out = preprocess_single(req.input_filename, req.output_filename)
    return {"status": "ok", "cleaned": out}

@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
