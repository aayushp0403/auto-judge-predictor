import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

TEXT_COL = "text_combined"
NUMERIC_COLS = ["solved_count", "contest_id"]

if __name__ == "__main__":
    model = joblib.load("models/lightgbm_cf_rating.txt")
    tfidf = joblib.load("models/tfidf_cf_rating.joblib")

    name = input("Enter problem name: ")
    tags = input("Enter tags (comma-separated): ")
    solved = float(input("Solved count: ") or 0)
    contest_id = float(input("Contest id (approx): ") or 0)

    df = pd.DataFrame([{
        "name": name,
        "tags": tags,
        "solved_count": solved,
        "contest_id": contest_id
    }])
    df[TEXT_COL] = df["name"].fillna("") + " " + df["tags"].fillna("")

    X_text = tfidf.transform(df[TEXT_COL])
    X_num = df[NUMERIC_COLS].fillna(0).astype(float).values
    X = hstack([X_text, X_num])

    pred = model.predict(X)[0]
    print("Predicted rating:", round(pred))
