from fastapi import FastAPI
import pandas as pd
from utils.data_loading import load_dataset

app = FastAPI()

@app.post("/")
async def get_server():
    return "Server running on port 80 !"

@app.post("/load")
async def load_data(path: str):
    df = load_dataset(path)
    return path
    
@app.post("/preprocess")
async def preprocess():
    pass

@app.post("/fit")
async def fit():
    pass

@app.post("/predict")
async def predict():
    pass