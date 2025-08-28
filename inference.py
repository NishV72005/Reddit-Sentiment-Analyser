import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load model + tokenizer from your trained folder
model_path = "./tinybert_reddit"   # <- your model folder
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Put model in evaluation mode
model.eval()

# 2. Example inputs (replace with your test data or load from CSV)
texts = [
    "I am very happy today!",
    "This is the worst day ever.",
    "I feel excited about the project."
]

# 3. Tokenized inputs
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 4. Run inference
with torch.no_grad():
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=-1).tolist()

# 5. Save results in CSV
df = pd.DataFrame({
    "text": texts,
    "prediction": predictions
})
df.to_csv("inference_results.csv", index=False)

print(" Predictions saved to inference_results.csv")
