import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Page config
st.set_page_config(page_title="📘 Persian Emotion Classifier")

# App title
st.title("📘 Persian Emotion Classifier")

# Try loading the model and tokenizer
try:
    st.write("🔄 Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained("parsbert-emotion")
    tokenizer = AutoTokenizer.from_pretrained("parsbert-emotion")
    labels = ['HAPPY', 'FEAR', 'SAD', 'HATE', 'ANGRY', 'SURPRISED', 'OTHER']
    st.success("✅ Model loaded successfully from local folder.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Text input
text = st.text_area("📝 Enter Persian text to analyze emotion:")

# Button click
if st.button("🔍 Analyze Emotion"):
    if not text.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        try:
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

            # Show results
            st.write("### 📊 Prediction Results:")
            for i, prob in enumerate(probs):
                st.write(f"{labels[i]}: {prob:.2f}")

            # Show bar chart
            st.bar_chart(probs)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
