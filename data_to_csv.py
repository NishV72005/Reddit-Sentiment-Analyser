# 1) Install deps (uncomment if running in a fresh environment)
# !pip install -q datasets pandas

from datasets import load_dataset
import pandas as pd

# --- Load GoEmotions ---
ds = load_dataset("go_emotions")  # splits: train/validation/test
label_names = ds["train"].features["labels"].feature.names  # list of 28 labels (27 + neutral)

# --- Convert each split to a DataFrame with readable labels ---
def split_to_df(split):
    df = split.to_pandas()
    # df["labels"] is a list of label indices; map to names and join by "|"
    df["labels"] = df["labels"].apply(lambda idxs: "|".join(label_names[i] for i in idxs))
    # keep only useful cols; rename 'text' if desired
    return df[["text", "labels"]].rename(columns={"text": "comment"})

full_df = pd.concat([split_to_df(ds["train"]),
                     split_to_df(ds["validation"]),
                     split_to_df(ds["test"])],
                    ignore_index=True)

# --- (Optional) Add one-hot columns for each emotion ---
for lab in label_names:
    full_df[lab] = full_df["labels"].apply(lambda s: int(lab in s.split("|")))

# --- Save single CSV ---
out_path = "goemotions_full.csv"
full_df.to_csv(out_path, index=False)
print(f"Saved: {out_path} | Rows: {len(full_df)} | Columns: {len(full_df.columns)}")
