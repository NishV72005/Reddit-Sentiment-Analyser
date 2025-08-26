import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)

# =====================
# 1. Load & preprocess
# =====================
df = pd.read_csv("reddit_comments.csv")

# Drop rows with missing labels
df = df.dropna(subset=["comment", "final_label"])

# Handle multi-label rows: take the first label only
df["final_label"] = df["final_label"].apply(lambda x: x.split(",")[0].strip())

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["final_label"])
num_labels = len(le.classes_)

print("Classes:", list(le.classes_))
print("Counts:", df["label"].value_counts().to_dict())

# Train/validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["comment"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42, stratify=df["label"]
)

# =====================
# 2. Dataset class
# =====================
class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# =====================
# 3. Tokenizer & datasets
# =====================
tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/bert-tiny")

train_dataset = RedditDataset(train_texts, train_labels, tokenizer)
val_dataset = RedditDataset(val_texts, val_labels, tokenizer)

# =====================
# 4. Model
# =====================
model = BertForSequenceClassification.from_pretrained(
    "prajjwal1/bert-tiny", num_labels=num_labels
)

# =====================
# 5. Class Weights
# =====================
class_counts = np.bincount(train_labels)
total_samples = len(train_labels)
weights = total_samples / (num_labels * class_counts)
class_weights = torch.tensor(weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

print("Class weights:", class_weights)

# =====================
# 6. Custom Trainer with weighted loss
# =====================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# =====================
# 7. Metrics
# =====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# =====================
# 8. TrainingArguments
# =====================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False
)

# =====================
# 9. Trainer
# =====================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# =====================
# 10. Train & Evaluate
# =====================
trainer.train()
metrics = trainer.evaluate()
print("Validation metrics:", metrics)
