import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# import nltk
# nltk.download('stopwords')

# Load dataset
dataset_path = r'/Users/rushiakbari/Desktop/Rushi/AI_Lab/AI_Project/fake-news/train.csv'
news_dataset = pd.read_csv(dataset_path)

# Display basic info
print("Dataset Shape:", news_dataset.shape)
print(news_dataset.head())
print(news_dataset.info())

# Fill missing values
news_dataset.fillna('', inplace=True)

# Combine 'author' and 'title' into a single 'content' column
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Separate features and labels
X = news_dataset['content']
Y = news_dataset['label']

# Preprocessing: Clean, tokenize, remove stopwords, and stem
stemmer = PorterStemmer()
english_stopwords = set(stopwords.words('english'))

def preprocess_text(content):
    content = re.sub('[^a-zA-Z]', ' ', content).lower()
    words = content.split()
    stemmed = [stemmer.stem(word) for word in words if word not in english_stopwords]
    return ' '.join(stemmed)

X = X.apply(preprocess_text)

# Data visualization: Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=news_dataset)
plt.title('Fake vs Real News Distribution')
plt.xlabel('Label (0: Real, 1: Fake)')
plt.ylabel('Count')
plt.show()

# Data visualization: Word cloud
all_words = ' '.join(X)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Common Words in News Content")
plt.show()

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Model training
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, Y_train)

# Evaluate on training data
train_preds = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_preds)
print("\nTraining Accuracy:", train_accuracy)

# Evaluate on test data
test_preds = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_preds)
print("Testing Accuracy:", test_accuracy)

# Classification report
print("\nClassification Report on Test Set:\n")
print(classification_report(Y_test, test_preds))

# Confusion matrix
cm = confusion_matrix(Y_test, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Cross-validation to check overfitting/underfitting
cv_scores = cross_val_score(model, X, Y, cv=5)
print("Cross-validation Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# Predict a single sample from test set
sample = X_test[3]
sample_prediction = model.predict(sample)

print("\nSample Prediction:", "Real News" if sample_prediction[0] == 0 else "Fake News")
print("Actual Label:", Y_test.iloc[3])