import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv("goemotions_clean.csv")

if "final_label" not in df.columns:
    raise ValueError(" 'final_label' column not found. Did you run merge.py first?")

# 2. Encode labels
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["final_label"])

# 3. Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["comment"].tolist(), df["label_id"].tolist(), test_size=0.2, random_state=42
)

# 4. Tokenizer
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# 5. Convert to Dataset
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels
})

val_dataset = Dataset.from_dict({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": val_labels
})

# 6. Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_encoder.classes_)
)

# 7. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    save_steps=500,   # older versions use save_steps not save_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False
)


# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,   # <-- Added validation dataset
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Manual evaluation after training
predictions, labels, _ = trainer.predict(val_dataset)
preds = np.argmax(predictions, axis=1)

from sklearn.metrics import accuracy_score, classification_report
print("Validation Accuracy:", accuracy_score(labels, preds))
print(classification_report(labels, preds, target_names=label_encoder.classes_))

# 10. Train
trainer.train()

# 11. Evaluate on validation set
results = trainer.evaluate()
print(f"Validation Accuracy: {results['eval_accuracy']:.4f}")

# 12. Save model + label encoder
trainer.save_model("./tinybert_reddit")
joblib.dump(label_encoder, "label_encoder.pkl")

print(" Training complete! Model saved to ./tinybert_reddit")

