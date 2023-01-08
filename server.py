from fastapi import FastAPI
import pandas as pd
from xgboost import XGBClassifier

from utils.data_loading import load_dataset
from utils.training import fit_predict

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
async def fit(data_path: str, target_variable: str):
    df = load_dataset(data_path)

    y_category_target = df[target_variable]
    x_input = df.drop([target_variable], axis=1)


    xgbc = XGBClassifier()

    report = fit_predict(x_input, y_category_target, 0.2, 42, xgbc)
    return report

@app.post("/predict")
async def predict():
    pass