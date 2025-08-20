import pandas as pd

# Load dataset
df = pd.read_csv("goemotions_full.csv")

# Emotion merge mapping
emotion_mapping = {
    "remorse": "Sadness",
    "sadness": "Sadness",
    "grief": "Sadness",

    "amusement": "Excitement",
    "excitement": "Excitement",
    "joy": "Excitement",
    "love": "Excitement",

    "fear": "Fear",
    "nervousness": "Fear",

    "anger": "Anger",
    "annoyance": "Anger",
    "disapproval": "Anger",
    "disgust": "Anger",

    "optimism": "Optimism",
    "pride": "Optimism",
    "relief": "Optimism",

    # keep these as-is
    "curiosity": "Others",
    "desire": "Others",
    "admiration": "Others",
    "realization": "Others",
    "confusion": "Others",
    "caring": "Others",
    "neutral": "Neutral"
}

# Merge labels row-wise
def merge_labels(row):
    active = [emo for emo in emotion_mapping.keys() if row.get(emo, 0) == 1]
    if not active:
        return "Neutral"
    merged = list({emotion_mapping[a] for a in active})
    return ", ".join(merged)  # keep multiple if present

df["final_label"] = df.apply(merge_labels, axis=1)

# Keep only relevant columns
df_clean = df[["comment", "final_label"]]

# Save cleaned dataset
df_clean.to_csv("goemotions_clean.csv", index=False)

print(" Saved cleaned dataset as goemotions_clean.csv")
print(df_clean.head(10))
