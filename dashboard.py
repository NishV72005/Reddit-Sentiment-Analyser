import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import plotly.express as px


# Load Model & Tokenizer

MODEL_NAME = "prajjwal1/bert-tiny"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained("./saved_model") 
model.eval()

# Defined label mapping (adjust according to your training)
label_map = {
    0: "Joy",
    1: "Sadness",
    2: "Anger",
    3: "Fear",
    4: "Neutral",
    5: "Others"
}


st.title(" Reddit Sentiment Analyzer Dashboard")
st.write("Enter a Reddit comment below and get real-time sentiment prediction.")

# User input
user_input = st.text_area(" Enter Reddit Comment:", "")

if st.button("Analyze"):
    if user_input.strip() != "":
        # Tokenized input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)

        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

       
        df = pd.DataFrame({
            "Sentiment": [label_map[i] for i in range(len(probs))],
            "Probability": probs.numpy()
        }).sort_values("Probability", ascending=False)

        # Display prediction
        st.subheader(f" Predicted Sentiment: **{df.iloc[0]['Sentiment']}**")

        # Plot probabilities
        fig = px.bar(df, x="Sentiment", y="Probability", color="Sentiment",
                     title="Prediction Probabilities", text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(" Please enter a comment before analyzing.")
