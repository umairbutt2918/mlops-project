import os
import tarfile
import urllib.request
import random
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models_utils import SimpleVectorizer, LogisticRegression, GradientBoosting
import numpy as np

DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_ARCHIVE = "aclImdb_v1.tar.gz"
EXTRACT_DIR = "aclImdb"

def download_and_extract():
    if not os.path.exists(DATA_ARCHIVE):
        print("Downloading dataset (this may take a while)...")
        urllib.request.urlretrieve(DATA_URL, DATA_ARCHIVE)
    else:
        print("Archive already downloaded.")
    if not os.path.exists(EXTRACT_DIR):
        print("Extracting...")
        with tarfile.open(DATA_ARCHIVE, "r:gz") as tar:
            tar.extractall()

def load_imdb_texts(base_path="aclImdb"):
    texts = []
    labels = []
    for split in ("train", "test"):
        for label in ("pos", "neg"):
            folder = os.path.join(base_path, split, label)
            if not os.path.exists(folder): continue
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
                    labels.append(1 if label == "pos" else 0)
    return texts, labels

def main():
    download_and_extract()
    texts, labels = load_imdb_texts(EXTRACT_DIR)
    print(f"Loaded {len(texts)} reviews.")

    if len(texts) > 20000:
        print("Subsampling to 20000 reviews for faster training.")
        idx = random.sample(range(len(texts)), 20000)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = SimpleVectorizer(max_features=10000)
    print("Fitting vectorizer...")
    X_train = vectorizer.fit_transform(X_train_texts)
    X_test = vectorizer.transform(X_test_texts)
    
    print("Training Logistic Regression...")
    logreg = LogisticRegression(learning_rate=0.1, max_iter=1000)
    logreg.fit(X_train, np.array(y_train))
    print("Acc LR:", accuracy_score(y_test, logreg.predict(X_test)))

    print("Training Gradient Boosting...")
    gb = GradientBoosting(n_estimators=30, learning_rate=0.1, max_depth=3, subsample=0.8)
    gb.fit(X_train, np.array(y_train))
    print("Acc GB:", accuracy_score(y_test, gb.predict(X_test)))

    with open("vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)
    with open("logistic_regression_model.pkl", "wb") as f: pickle.dump(logreg, f)
    with open("gradient_boosting_model.pkl", "wb") as f: pickle.dump(gb, f)
    print("Models saved.")

if __name__ == "__main__":
    main()
