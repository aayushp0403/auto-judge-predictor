import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import joblib

from features_baseline import make_features

DIFF_COL = "Difficulty"

def load_split(prefix):
    train = pd.read_csv(f"data/{prefix}_train.csv")
    val = pd.read_csv(f"data/{prefix}_val.csv")
    test = pd.read_csv(f"data/{prefix}_test.csv")
    return train, val, test

if __name__ == "__main__":
    train, val, test = load_split("leetcode")

    le = LabelEncoder()
    y_train = le.fit_transform(train[DIFF_COL])
    y_val = le.transform(val[DIFF_COL])
    y_test = le.transform(test[DIFF_COL])

    X_train, tfidf = make_features(train, fit_tfidf=True)
    X_val, _ = make_features(val, tfidf=tfidf)
    X_test, _ = make_features(test, tfidf=tfidf)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=3,
        n_estimators=2000
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, "models/lightgbm_leetcode.txt")
    joblib.dump(le, "models/label_encoder_leetcode.joblib")
    joblib.dump(tfidf, "models/tfidf_leetcode.joblib")
