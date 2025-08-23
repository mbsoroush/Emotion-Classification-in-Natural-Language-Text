# 📘 Persian Emotion Classification (NLP Project)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Transformers](https://img.shields.io/badge/Transformers-4.40+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

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

```


## 🚀 Usage

To run the application locally:

1. ✅ Open a terminal in the project folder  
2. ✅ Run the following command:  

   ```bash
   streamlit run app.py
   ```

## 🧹 Preprocessing Steps

Before training, the Persian text data went through several preprocessing steps:

1. ✅ **Text Normalization**  
   - Remove elongated letters (e.g., `خیییییییلی` → `خیلی`)  
   - Unify punctuation (convert Arabic to Persian variants, normalize question marks, etc.)  
   - Remove extra spaces and unwanted characters (like emojis or Latin words if present)  

2. ✅ **Tokenization**  
   - Use the ParsBERT tokenizer (`HooshvareLab/bert-base-parsbert-uncased`)  
   - Truncate or pad sequences to a maximum of **128 tokens**  

3. ✅ **Label Encoding**  
   - Convert categorical labels (HAPPY, SAD, etc.) into integer IDs  
   - Example: `{"HAPPY": 0, "FEAR": 1, "SAD": 2, ...}`  

4. ✅ **Dataset Splitting**  
   - Training set → **6,000 samples**  
   - Test set → **1,000 samples**  

---

📌 After preprocessing, the dataset was ready for input into the ParsBERT model.


## 🧠 Model Training Steps

The fine-tuning of ParsBERT was done on Kaggle with GPU acceleration.  
Below are the main steps:

1. ✅ **Model Selection**  
   - Base model: `HooshvareLab/bert-base-parsbert-uncased`  
   - Suitable for Persian NLP tasks  

2. ✅ **Training Environment**  
   - Kaggle GPU (Tesla **T4** or **P100**)  
   - Python 3.10, Transformers 4.40+  

3. ✅ **Hyperparameters**  
   - Epochs: **4**  
   - Batch size: **16**  
   - Learning rate: **2e-5**  
   - Max sequence length: **128**  

4. ✅ **Evaluation Metric**  
   - **Macro F1-score** (handles class imbalance better than accuracy)  

5. ✅ **Training Process**  
   - Train dataset: 6,000 samples  
   - Test dataset: 1,000 samples  
   - Saved the **best model checkpoint** at the end of training  

6. ✅ **Results**  
   - Final Macro F1-score: **~0.72**  
   - Good overall performance  
   - Some overlap between *ANGRY*, *HATE*, and *SAD* classes (expected due to semantic similarity)  

---

📌 After training, the best-performing model was exported to the folder `parsbert-emotion/` for use in the Streamlit app.


## 📊 Example Prediction Steps

Here’s how the model performs on a sample Persian input:

1. ✅ **User Input**  
من خیلی ناراحت هستم


2. ✅ **Model Processing**  
- Text is tokenized with ParsBERT tokenizer  
- Input is passed through the fine-tuned `parsbert-emotion` model  

3. ✅ **Prediction Output**  
Example probabilities:  
- 😢 **SAD** → `0.82`  
- 😠 **ANGRY** → `0.10`  
- ❓ **OTHER** → `0.05`  

4. ✅ **Visualization**  
- Streamlit displays a **bar chart** of all 7 emotion probabilities  
- The highest probability is shown as the predicted emotion  

---

📌 In this case, the model correctly classifies the sentence as expressing **Sadness**.


## 🧪 Troubleshooting

Common issues and their solutions when running the project:

1. ❌ **Error: `meta tensor` or corrupted model**  
   - Cause: Model files were incomplete or corrupted  
   - ✅ Solution:  
     - Re-extract the `parsbert-emotion/` folder  
     - Ensure it contains all required files:  
       (`pytorch_model.bin`, `config.json`, `tokenizer.json`, `vocab.txt`, etc.)  

2. ❌ **Blank Streamlit Page**  
   - Cause: Running with `python app.py` instead of Streamlit  
   - ✅ Solution:  
     - Always run with:  
       ```bash
       streamlit run app.py
       ```  

3. ❌ **NumPy Conversion Error**  
   - Cause: Attempting `.numpy()` directly on a GPU tensor  
   - ✅ Solution:  
     - Use `.cpu().numpy()` instead to safely move tensors to CPU  

4. ❌ **Model ID Not Found**  
   - Cause: `parsbert-emotion` is not available on Hugging Face Hub  
   - ✅ Solution:  
     - Use the **local folder name** when loading:  
       ```python
       AutoModelForSequenceClassification.from_pretrained("parsbert-emotion")
       ```  

---

📌 Following these steps should resolve most issues encountered when running the model or app.

## 🚀 Deployment Options

You can run the Persian Emotion Classifier both locally and online:

1. ✅ **Local Deployment**  
   - Run the app with:  
     ```bash
     streamlit run app.py
     ```  
   - Opens on `http://localhost:8501`  

2. ✅ **Online Deployment Options**  
   - **Streamlit Cloud** → Quick, free deployment for sharing apps  
   - **Hugging Face Spaces** → Free hosting with Streamlit or Gradio  

---

## 📦 Deliverables

The project includes the following components:

1. ✅ Fine-tuned model → `parsbert-emotion/`  
2. ✅ Training notebook → `emotions-classification-nlp.ipynb`  
3. ✅ Streamlit app → `app.py`  
4. ✅ Project report → `report.pdf`  
5. ✅ Dataset → [Emotions in Persian Texts](https://www.kaggle.com/datasets/mbsoroush/emotions-in-persian-texts)  

---

## 🏁 Conclusion

1. ✅ **Objective Achieved**  
   - Fine-tuned **ParsBERT** for Persian emotion classification  

2. ✅ **Key Features**  
   - Supports **7 emotion classes**  
   - Achieved **Macro F1 ~0.72**  
   - Integrated with an interactive **Streamlit app**  

3. ✅ **Impact**  
   - Demonstrates the power of transformer models in **low-resource languages** like Persian  
   - Provides a practical tool for text emotion analysis  

4. ✅ **Future Improvements**  
   - Expand dataset with more labeled examples  
   - Explore advanced transformer models (e.g., RoBERTa, mBERT)  
   - Deploy on cloud for public access  

---

📌 With this project, we showed that **transformer-based NLP models** can effectively handle **Persian emotion classification**, bridging the gap for practical AI applications in Persian language processing.


### ⚖️ License

This repository is provided **without an open-source license**.  
All rights reserved – you may **not** use, copy, modify, or distribute this code without explicit permission.  




