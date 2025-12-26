import pandas as pd
import joblib
import shap
from features_baseline import make_features

DIFF_COL = "Difficulty"

if __name__ == "__main__":
    model = joblib.load("models/lightgbm_leetcode.txt")
    tfidf = joblib.load("models/tfidf_leetcode.joblib")
    le = joblib.load("models/label_encoder_leetcode.joblib")

    test = pd.read_csv("data/leetcode_test.csv")
    X_test, _ = make_features(test, tfidf=tfidf)

    # Use the generic Explainer wrapper
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)   # shap_values is a shap._explanation.Explanation

    print("Classes:", le.classes_)

    # Plot summary for one class (e.g., index 0)
    shap.plots.beeswarm(shap_values[:, :, 0], show=True)
