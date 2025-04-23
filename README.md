📰 Fake News Detection using Machine Learning

Detect whether a news article is real or fake using Natural Language Processing and Machine Learning techniques. This project applies text preprocessing, TF-IDF vectorization, and a Logistic Regression model to predict news authenticity.

🔍 Project Overview
Fake news poses a serious challenge to modern society. This project leverages supervised machine learning to classify news articles as real or fake based on their title and author using the following steps:

Data Cleaning and Preprocessing (Tokenization, Stopwords Removal, Stemming)

Exploratory Data Analysis and Visualization (WordClouds, Count Plots)

Feature Engineering with TF-IDF

Model Training using Logistic Regression

Evaluation using Accuracy, Confusion Matrix, and Classification Report

Cross-validation for model robustness

📁 Dataset
The dataset used in this project is from Kaggle's Fake News dataset.

Features:

title: The title of the news article

author: The author of the news article

text: The text body of the article (not used in this version)

label: 1 for Fake, 0 for Real

🧠 Model

Model	Accuracy (Test)	Accuracy (Train)	Cross-Validation (Mean)
Logistic Regression	✅ ~96%	✅ ~98%	✅ ~95%
The model is trained using LogisticRegression from sklearn with a TfidfVectorizer on the combined and cleaned author + title content.

📊 Visualizations
1. Label Distribution
Shows the count of Real vs Fake news articles.

2. Word Cloud
Visualizes the most frequent words used in the news dataset after cleaning and stemming.

🛠️ Requirements
Install the required libraries with:

bash
Copy
Edit
pip install -r requirements.txt
Main Libraries:
numpy

pandas

matplotlib

seaborn

nltk

scikit-learn

wordcloud

Note: You may need to download NLTK stopwords manually:

python
Copy
Edit
import nltk
nltk.download('stopwords')
🚀 Run the Project
Clone the repository and run the main script:

bash
Copy
Edit
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
python main.py
✅ Sample Prediction
The script also includes a prediction for a sample news article from the test set, showing both the predicted and actual label.

📌 Future Improvements
Use the full article text for better accuracy

Experiment with other models (e.g., Random Forest, SVM, BERT)

Build a Flask or Streamlit web app for real-time detection

📄 License
This project is licensed under the MIT License. Feel free to use and modify for educational or commercial purposes.

🙌 Acknowledgements
Kaggle Fake News Dataset

Scikit-learn and NLTK documentation

Visualization inspiration from Seaborn & WordCloud

