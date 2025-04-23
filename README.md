# 📰 Fake News Detection using Machine Learning  
---

## 🚀 Overview  
---
This project is a complete pipeline for detecting **fake news articles** using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. Built in Python, it leverages `scikit-learn`, `NLTK`, and visualization libraries to process news data and classify it as **real** or **fake** with high accuracy.

The goal of this project is to build a reliable text-based classifier that can distinguish fake news using:

- 🧹 **Text Preprocessing** (Cleaning, Tokenization, Stopwords Removal, Stemming)  
- ✨ **TF-IDF Vectorization** for converting text to numerical features  
- 🧠 **Logistic Regression Model** for classification  
- 📊 **Visual Explorations** (Word Clouds, Count Plots)  
- 📈 **Model Evaluation** (Accuracy, Confusion Matrix, Classification Report)  
- 🔁 **Cross-validation** to test model robustness  

---

## 📁 Dataset  
---
Dataset used: [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)

**Features:**

- 📝 `title` — News headline  
- 👤 `author` — Article author  
- 📰 `text` — Full news article (not used in this version)  
- 🏷️ `label` — 1 = Fake, 0 = Real  

---

## 📚 Tech Stack  
---
- **Language:** Python  
- **Libraries & Tools:**  
  - `numpy`, `pandas`  
  - `matplotlib`, `seaborn`, `wordcloud`  
  - `nltk` for text processing  
  - `scikit-learn` for ML & evaluation  

---

## 📊 Visualizations  
---
- 🔢 **Class Distribution** – Count of real vs fake news articles  
- ☁️ **Word Cloud** – Most frequent words in the cleaned dataset  

---

## 🧠 Model Performance  
---
| Metric                | Score     |
|-----------------------|-----------|
| ✅ **Test Accuracy**    | ~96%      |
| 🧪 **Train Accuracy**   | ~98%      |
| 🔁 **Cross-Validation** | ~95%      |

Model used: **Logistic Regression** with **TF-IDF vectorization** on cleaned news title + author content.

---

## ⚙️ Installation  
---
1. Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
