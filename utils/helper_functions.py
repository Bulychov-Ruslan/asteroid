import pandas as pd

def load_dataset(file_path):
    return pd.read_csv(file_path, low_memory=False)
