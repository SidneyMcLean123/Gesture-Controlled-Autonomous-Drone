import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("raw_gesture_data.csv")

y = df.iloc[:, -1].values

encoder = LabelEncoder()
encoder.fit(y)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)