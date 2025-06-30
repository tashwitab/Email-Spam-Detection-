# Email-Spam-Detection-
A machine learning-based model to classify emails as spam or not spam using NLP and classification algorithms.

# 📧 Email Spam Detection using Machine Learning

A machine learning project to classify emails as **spam** or **not spam** using Natural Language Processing (NLP) techniques and classification algorithms. This project demonstrates how textual data can be converted into numerical form and used to build an effective predictive model.

---

## 🚀 Features

- Data preprocessing and cleaning of raw email text
- Feature extraction using TF-IDF Vectorizer
- Model training using Naive Bayes, SVM, and Logistic Regression
- Performance metrics: Accuracy, Precision, Recall, F1-score
- Interactive email prediction (optional in notebook)

---

## 📦 Tech Stack

- **Languages**: Python
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **NLP Tools**: TF-IDF Vectorizer, NLTK
- **Modeling**: Multinomial Naive Bayes, SVM, Logistic Regression

---

## 📊 Dataset

- Source: [Kaggle – Spam Detection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format: SMS or email texts with labels (`spam` or `ham`)

---

## 🧪 Workflow

1. **Data Cleaning**  
   - Lowercasing, punctuation removal, stopwords removal, stemming

2. **Feature Extraction**  
   - Convert text to numerical vectors using TF-IDF

3. **Model Training & Evaluation**  
   - Train-test split, model fitting, metric evaluation

4. **Prediction**  
   - Real-time email text classification

---

## 📈 Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Naive Bayes        | 98.7%    | 98.2%     | 97.9%  | 98.0%    |
| Logistic Regression| 97.5%    | 96.8%     | 95.6%  | 96.2%    |
| SVM                | 97.2%    | 96.4%     | 95.1%  | 95.7%    |

---
## 🤝 Contributors

- **Tashwita Bhirud** – Electronics & Telecommunication + AIML
- **Shruti Munde** - Electronics & Telecommunication + AIML

## 🖥️ Sample Prediction

```python
Input: "Congratulations! You have won a free cruise to Bahamas!"
Output: Spam

Input: "Please find the meeting agenda attached."
Output: Not Spam


