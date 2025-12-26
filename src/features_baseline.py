import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

TEXT_COL = "text_combined"
NUMERIC_COLS = []   # no numeric features

def add_text_combined(df):
    # use Title + Topics if available
    df = df.copy()
    df[TEXT_COL] = (
        df["Title"].fillna("") + " " +
        df["Topics"].fillna("")
    )
    return df

def make_features(df, tfidf=None, fit_tfidf=False):
    df = add_text_combined(df)
    text_data = df[TEXT_COL].fillna("")

    if tfidf is None and fit_tfidf:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=3
        )
        X_text = tfidf.fit_transform(text_data)
    else:
        X_text = tfidf.transform(text_data)

    if NUMERIC_COLS:
        X_num = df[NUMERIC_COLS].fillna(0.0).astype(float).values
        X = hstack([X_text, X_num])
    else:
        X = X_text

    return X, tfidf

if __name__ == "__main__":
    train = pd.read_csv("data/leetcode_train.csv")
    X_train, tfidf = make_features(train, fit_tfidf=True)
    joblib.dump(tfidf, "models/tfidf_leetcode.joblib")
    print("Train shape:", X_train.shape)
