📰 Fake News Detection using Machine Learning
Detect whether a news article is real or fake using Natural Language Processing and Machine Learning. This project utilizes text preprocessing, TF-IDF vectorization, and Logistic Regression to classify news authenticity.

🔍 Project Overview
Fake news is a growing problem in the digital age. This project uses supervised machine learning to predict whether a news article is fake or real based on its title and author. Key steps:

🧹 Data Cleaning & Preprocessing (Tokenization, Stopword Removal, Stemming)

📊 Exploratory Data Analysis (WordCloud, Count Plots)

🧾 Feature Extraction using TF-IDF

🤖 Model Training with Logistic Regression

📈 Performance Evaluation (Accuracy, Confusion Matrix, Classification Report)

🔁 Cross-validation for robustness

📁 Dataset
This project uses the Kaggle Fake News Dataset.

Features:

📝 title: Title of the news article

👤 author: Author of the article

📄 text: Full body of the news (not used in this version)

🏷️ label: 1 for Fake, 0 for Real

🧠 Model Performance

Model	✅ Test Accuracy	🧪 Train Accuracy	🔁 Cross-Validation
Logistic Regression	~96%	~98%	~95%
Model: LogisticRegression from scikit-learn
Features: TF-IDF vectorized content (author + title)

📊 Visualizations
1️⃣ Label Distribution
Visualizes the count of fake vs real news articles.

2️⃣ Word Cloud
Highlights the most common words after preprocessing.

🛠️ Requirements
Install the dependencies using:

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

📌 Note:
Download NLTK stopwords if not already available:

python
Copy
Edit
import nltk
nltk.download('stopwords')
🚀 Run the Project
Clone the repo and run the script:

bash
Copy
Edit
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
python main.py
✅ Sample Prediction
The script predicts whether a randomly selected article from the test set is fake or real and displays both:

🔮 Predicted Label: Fake/Real

🎯 Actual Label: Fake/Real

📌 Future Improvements
📰 Use full article text for deeper context

🔍 Try advanced models (Random Forest, SVM, BERT)

🌐 Build a real-time web app (Flask / Streamlit)

📄 License
Licensed under the MIT License. Free to use and modify for personal, educational, or commercial purposes.

🙌 Acknowledgements
Kaggle Fake News Dataset

scikit-learn, nltk documentation

Visualizations inspired by seaborn, wordcloud
