import pandas as pd
from pandas_profiling import ProfileReport

def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def get_profiling(df):
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("data_report.html")
    return profile