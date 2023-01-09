from fastapi import FastAPI
import pandas as pd
from xgboost import XGBClassifier
import os

from utils.training import fit_predict
from utils.data_loading import load_dataset, get_profiling
from utils.preprocessing import preprocessing

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
    
@app.post("/fit")
async def fit(data_path: str, target_variable: str):
    df = load_dataset(data_path)

    
    (x_input, y_category_target) = preprocessing(df, target_variable, completion_rate=0.4)

    xgbc = XGBClassifier()

    report = fit_predict(x_input, y_category_target, 0.2, 42, xgbc)
    return report