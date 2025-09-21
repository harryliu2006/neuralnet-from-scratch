
import os
import zipfile
import numpy as np
import pandas as pd

from neuralnet import (
    train_val_split, standardize_train_val,
    train_early_stop, accuracy
)


zip_path = os.path.expanduser("~/Downloads/breastCancer.zip")
extract_dir = "data"
csv_name = "data.csv"

def prepare_data():
   
    os.makedirs(extract_dir, exist_ok=True)
    csv_path = os.path.join(extract_dir, csv_name)
    if not os.path.exists(csv_path) and os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 32"], errors="ignore")
    y = (df["diagnosis"].values.reshape(1, -1) == "M").astype(np.float64)
    X = df.drop(columns=["diagnosis"]).values.T.astype(np.float64)
    return X, y

def main():
    X, Y = prepare_data()
    X_train, Y_train, X_val, Y_val = train_val_split(X, Y, val_ratio=0.2, seed=42)
    X_train, X_val = standardize_train_val(X_train, X_val)

    n_in = X_train.shape[0]          
    layer_sizes = [n_in, 16, 8, 1]   

    W, B = train_early_stop(
        layer_sizes,
        X_train, Y_train,
        X_val,   Y_val,
        epochs=5000, lr=0.1, patience=20, min_delta=1e-3, print_every=100
    )

    tr = accuracy(W, B, X_train, Y_train)
    va = accuracy(W, B, X_val,   Y_val)
    print(f"Final → train_acc={tr:.3f}  val_acc={va:.3f}")

if __name__ == "__main__":
    main()
