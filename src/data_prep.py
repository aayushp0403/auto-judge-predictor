import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_COL = "Title"
DIFF_COL = "Difficulty"

def load_leetcode(path="data/leetcode_problems.csv"):
    df = pd.read_csv(path)
    # Drop rows with missing critical info
    df = df.dropna(subset=[TEXT_COL, DIFF_COL])
    return df

def split_save(df, prefix):
    train, temp = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df[DIFF_COL]
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp[DIFF_COL]
    )
    train.to_csv(f"data/{prefix}_train.csv", index=False)
    val.to_csv(f"data/{prefix}_val.csv", index=False)
    test.to_csv(f"data/{prefix}_test.csv", index=False)

if __name__ == "__main__":
    lc = load_leetcode()
    split_save(lc, "leetcode")
