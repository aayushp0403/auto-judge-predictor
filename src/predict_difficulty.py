import joblib
import pandas as pd
from features_baseline import make_features

DIFF_COL = "Difficulty"

if __name__ == "__main__":
    model = joblib.load("models/lightgbm_leetcode.txt")
    tfidf = joblib.load("models/tfidf_leetcode.joblib")
    le = joblib.load("models/label_encoder_leetcode.joblib")

    title = input("Enter problem title: ")
    topics = input("Enter topics/tags (comma-separated): ")
    acc_rate = float(input("Acceptance rate (%): ") or 50)
    likes = float(input("Likes: ") or 0)
    dislikes = float(input("Dislikes: ") or 0)

    df = pd.DataFrame([{
        "Title": title,
        "Topics": topics,
        "Acceptance Rate (%)": acc_rate,
        "Likes": likes,
        "Dislikes": dislikes
    }])

    X, _ = make_features(df, tfidf=tfidf)
    pred = model.predict(X)
    label = le.inverse_transform(pred)[0]

    print("Predicted difficulty:", label)
