# Emotion-Classification-in-Natural-Language-Text

📘 Persian Emotion Classification (NLP Project)








🔰 Overview

This project is an NLP-based Persian Emotion Classifier built with ParsBERT
.
It takes informal Persian text (e.g., from social media) and predicts its emotion across 7 categories:

😃 HAPPY

😨 FEAR

😢 SAD

😡 HATE

😠 ANGRY

😲 SURPRISED

❓ OTHER

The project consists of two main parts:

Model Training → Fine-tuning ParsBERT on a Persian emotion dataset using Kaggle GPU.

Streamlit App → An interactive interface where users enter text and see emotion predictions in real-time.

📂 Project Structure
📁 persian-emotion-classifier/
│── 📄 app.py                     # Streamlit app
│── 📄 requirements.txt           # Python dependencies
│── 📄 README.md                  # Project documentation
│── 📄 report.pdf                 # Final project report
│── 📁 parsbert-emotion/          # Fine-tuned model & tokenizer
│── 📁 notebooks/
│    └── emotions-classification-nlp.ipynb   # Kaggle training notebook

📦 Dataset

The dataset is available publicly on Kaggle:
👉 Emotions in Persian Texts

6,000 training samples

1,000 test samples

Each record contains:

A short informal Persian text (e.g., from social media)

A labeled emotion (one of 7 categories)

The dataset includes noisy, real-world text such as:

Emojis

Elongated words (خیییییلی → خیلی)

Mixed characters

⚙️ Installation & Setup
1. Clone Repository
git clone https://github.com/your-username/persian-emotion-classifier.git
cd persian-emotion-classifier

2. Create Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Run Streamlit App
streamlit run app.py

🧹 Preprocessing Steps

Normalize Persian text (fix elongated letters, remove Latin chars, unify punctuation)

Label encoding of emotions

Tokenization with ParsBERT tokenizer

Limit sequence length to 128 tokens

🧠 Model Training

Base model: HooshvareLab/bert-base-parsbert-uncased

Training platform: Kaggle GPU (Tesla T4/P100)

Parameters:

Epochs: 4

Batch size: 16

Max sequence length: 128

Evaluation metric: Macro F1-score

Results:

Macro F1-score: ~0.72

Minor confusion between ANGRY, HATE, and SAD (expected overlap)

🖥️ Streamlit App

The app allows users to type Persian text and see predicted emotions with confidence scores.

Example:

Input:

من خیلی ناراحت هستم


Output:

SAD: 0.82
ANGRY: 0.10
OTHER: 0.05


Also displays a bar chart of probabilities.

📊 Results Snapshot
Emotion	Example Sentence	Prediction
HAPPY 😃	من امروز خیلی خوشحال هستم	HAPPY
SAD 😢	من خیلی ناراحت و تنها هستم	SAD
ANGRY 😠	از دست تو خیلی عصبانی‌ام	ANGRY
SURPRISED 😲	باورم نمی‌شه این اتفاق افتاد!	SURPRISED
🧪 Troubleshooting

Error: meta tensor
Fix: Ensure parsbert-emotion/ folder has all model files (pytorch_model.bin, config.json, etc.).

Blank Streamlit page:
Run with streamlit run app.py (not python app.py).

NumPy conversion error:
Use .cpu().numpy() instead of .numpy() directly.

🚀 Deployment Options

Local: Run with streamlit run app.py

Online:

Deploy to Streamlit Cloud

Deploy to Hugging Face Spaces

📦 Deliverables

✅ Fine-tuned model (parsbert-emotion/)

✅ Training notebook (.ipynb)

✅ Streamlit app (app.py)

✅ Project report (report.pdf)

✅ Dataset (hosted on Kaggle here
)

🏁 Conclusion

This project demonstrates how ParsBERT can be effectively fine-tuned for Persian NLP tasks.
The combination of a transformer model and a Streamlit app makes it practical for real-world Persian emotion detection.
