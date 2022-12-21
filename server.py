from fastapi import FastAPI
import pandas as pd

app = FastAPI()


@app.get("/")
async def load():
    return "Server running on port 3000 !"

@app.post("/preprocess")
async def preprocess():
    pass

@app.post("/fit")
async def fit():
    pass

@app.post("/predict")
async def predict():
    pass