# 📘 Persian Emotion Classification (NLP Project)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Transformers](https://img.shields.io/badge/Transformers-4.40+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🔰 Overview
This project is an **NLP-based Persian Emotion Classifier** built with [ParsBERT](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased).  
It predicts emotions from Persian text across **7 categories**:

- 😃 HAPPY  
- 😨 FEAR  
- 😢 SAD  
- 😡 HATE  
- 😠 ANGRY  
- 😲 SURPRISED  
- ❓ OTHER  

The project has two main parts:
1. **Model Training** → Fine-tuning ParsBERT on a Persian emotion dataset using Kaggle GPU.  
2. **Streamlit App** → An interactive interface where users enter text and get predictions in real-time.

---

## 📦 Dataset
We used the dataset hosted on Kaggle:  
👉 [Emotions in Persian Texts](https://www.kaggle.com/datasets/mbsoroush/emotions-in-persian-texts)

- **6,000** training samples  
- **1,000** test samples  
- Each record contains:
  - A Persian text  
  - An emotion label (one of 7 categories)  

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash

git clone https://github.com/your-username/persian-emotion-classifier.git
cd persian-emotion-classifier
pip install -r requirements.txt
\```


## 🚀 Usage

To run the application locally:

1. ✅ Open a terminal in the project folder  
2. ✅ Run the following command:  

   ```bash
   streamlit run app.py
