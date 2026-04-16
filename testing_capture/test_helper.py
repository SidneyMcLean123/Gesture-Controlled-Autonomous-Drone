# test_helpers.py
import numpy as np
import pandas as pd

def load_sample_by_label(label, csv_path="test_data.csv"):
    """Returns the first matching landmark array for a given gesture label."""
    df = pd.read_csv(csv_path, header=None)
    
    # Last column is the label
    match = df[df.iloc[:, -1] == label]
    
    if match.empty:
        raise ValueError(f"No samples found for label: {label}")
    
    # Take first match, drop label column, return as numpy array
    landmarks = match.iloc[0, :-1].values.astype(float).reshape(1, -1)
    return landmarks

def load_all_samples(csv_path="test_data.csv"):
    """Returns all landmark arrays and their labels as parallel lists."""
    df = pd.read_csv(csv_path, header=None)
    labels = df.iloc[:, -1].values
    landmarks = df.iloc[:, :-1].values.astype(float)
    return landmarks, labels