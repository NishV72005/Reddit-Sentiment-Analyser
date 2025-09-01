import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# ======================
# Load dataset
# ======================
df = pd.read_csv("goemotions_full.csv")

# Make sure label column exists
if "final_label" not in df.columns:
    raise ValueError(" 'final_label' column not found in CSV. Please run merge.py first.")

# Encode labels
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["final_label"])

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["comment"].tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=42
)


model_name = "prajjwal1/bert-tiny"   # very small & fast BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


num_labels = len(label_encoder.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",      # used eval at each epoch
    save_strategy="epoch",      
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"  
)


def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, f1_score
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()


save_dir = "./tinybert_reddit"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Save label mapping
import json
with open(f"{save_dir}/label_encoder.json", "w") as f:
    json.dump({i: label for i, label in enumerate(label_encoder.classes_)}, f)

print(" Training complete. Model saved at", save_dir)
