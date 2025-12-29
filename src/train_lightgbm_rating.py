import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

TEXT_COL = "text_combined"
TARGET_COL = "rating"
NUMERIC_COLS = ["solved_count", "contest_id"]

df = pd.read_csv("data/codeforces_problems.csv")
df = df.dropna(subset=[TARGET_COL])

# build combined text
df[TEXT_COL] = df["name"].fillna("") + " " + df["tags"].fillna("")

train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=3)
X_train_text = tfidf.fit_transform(train[TEXT_COL])
X_val_text = tfidf.transform(val[TEXT_COL])
X_test_text = tfidf.transform(test[TEXT_COL])

X_train_num = train[NUMERIC_COLS].fillna(0).astype(float).values
X_val_num = val[NUMERIC_COLS].fillna(0).astype(float).values
X_test_num = test[NUMERIC_COLS].fillna(0).astype(float).values

X_train = hstack([X_train_text, X_train_num])
X_val = hstack([X_val_text, X_val_num])
X_test = hstack([X_test_text, X_test_num])

y_train = train[TARGET_COL].astype(float).values
y_val = val[TARGET_COL].astype(float).values
y_test = test[TARGET_COL].astype(float).values

model = lgb.LGBMRegressor(
    objective="regression",
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
    eval_metric="l2",
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
print("Test MAE:", mae)

joblib.dump(model, "models/lightgbm_cf_rating.txt")
joblib.dump(tfidf, "models/tfidf_cf_rating.joblib")
