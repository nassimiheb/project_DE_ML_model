from fastapi import FastAPI
import pandas as pd
from utils.data_loading import load_dataset, get_profiling
import os

app = FastAPI()

@app.post("/")
async def get_server():
    return "Server running on port 80 !"

@app.post("/load")
async def load_data(path: str):
    df = load_dataset(path)
    profile = get_profiling(df)

    profile.to_file('profile.html')

    return str(os.path.dirname(os.path.realpath(__file__))) + "/profile.html"
    
@app.post("/preprocess")
async def preprocess():
    pass

@app.post("/fit")
async def fit(data_path: str, target_variable: str):
    pass

@app.post("/predict")
async def predict():
    pass