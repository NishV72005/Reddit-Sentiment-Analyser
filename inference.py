# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd


model_path = "tinybert_reddit"   
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()


df = pd.read_csv("goemotions_clean.csv")   

# Splited multi-label entries like "Anger, Neutral" → ["Anger", "Neutral"]
all_labels = set()
for item in df["final_label"].dropna():
    for lbl in str(item).split(","):
        all_labels.add(lbl.strip())

emotion_classes = sorted(list(all_labels))


if len(emotion_classes) != model.config.num_labels:
    raise ValueError(
        f"Label count ({len(emotion_classes)}) does not match model outputs ({model.config.num_labels}). "
        "Please check label extraction or training setup."
    )


def predict_emotion(text: str) -> str:
    """Predict the single most probable emotion for a given text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=1).item()

    return emotion_classes[predicted_class]

# Example Usage

if __name__ == "__main__":
    samples = [
        "I am really happy with how my day went!",
        "This is the worst experience I’ve ever had.",
        "Feeling anxious about tomorrow’s exam.",
        "Wow, that news was so exciting!",
        "I don’t really care anymore."
    ]
    
    print("Available Emotion Classes:", emotion_classes, "\n")
    
    for text in samples:
        label = predict_emotion(text)
        print(f"Text: {text}")
        print(f"Predicted Emotion: {label}\n")
