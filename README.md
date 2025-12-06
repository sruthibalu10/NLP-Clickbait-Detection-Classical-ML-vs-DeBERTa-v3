# NLP-Clickbait-Detection-Classical-ML-vs-DeBERTa-v3
This project analyzes clickbait using classical ML models and a fine-tuned DeBERTa-v3 transformer, incorporating preprocessing, sentiment analysis, and feature engineering. Model comparison shows DeBERTa-v3 best captures contextual and emotional cues for accurate clickbait detection.

## 📘 Overview
Clickbait headlines are designed to attract attention at the cost of informative quality. Automated detection helps improve content moderation, ranking, and credibility evaluation.  
This project investigates:

- How classical ML models perform with hand-crafted text features  
- How a transformer model like DeBERTa-v3 performs when fine-tuned for classification  
- Which approach generalizes better and why  

The goal is to provide a comparative understanding of modern NLP techniques.

## 🧹 Data Preprocessing
The NLP pipeline includes:

- Text normalization (lowercasing, punctuation removal, token cleaning)
- Stopword handling
- Vectorization using:
  - **Bag-of-Words**
  - **TF-IDF** (unigrams + bigrams)
- Label mapping for multi-class clickbait detection

Visualizations (word clouds, class distributions, etc.) support the EDA process.

## 🧠 Models Compared

### **Classical ML Models**
- Logistic Regression  
- Linear SVM (LinearSVC)  
- Random Forest  
- Naive Bayes  
- XGBoost  

These models were trained using both **CountVectorizer** and **TF-IDF** representations.

### **Transformer Model**
- **DeBERTa-v3 Base**

The transformer model was fine-tuned on combined title + text fields, using HuggingFace Transformers.

## 📊 Evaluation
Models were compared based on:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrices  

Observations:

- TF-IDF consistently outperformed raw count vectors for classical ML models.
- Logistic Regression and LinearSVC provided the strongest classical baselines.
- DeBERTa-v3 delivered significantly improved performance due to contextual embeddings.

## 🏁 Key Findings
- Classical ML models perform well on structured text representations but plateau in complex semantic understanding.
- DeBERTa-v3 captures contextual nuances that classical models cannot.
- The transformer model provides superior classification accuracy and robustness.
