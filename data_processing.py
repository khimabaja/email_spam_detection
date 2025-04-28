import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import joblib

# Download stopwords if not already present
# nltk.download('stopwords')

# Load CSV file
CSV_PATH = 'datasets/enron_spam_data.csv'  # update with your file path
df = pd.read_csv(CSV_PATH)

# print(df.head())
# print(df.info())

# Normalize labels if needed (e.g., "spam"/"ham" → 1/0)
df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})  # only if labels are strings
# print(df.head())
# print(df.info())

# # Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# # Apply preprocessing
df['clean_text'] = df['Message'].apply(preprocess_text)

# print(df.head())

# # Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# print(X_test.head())
# print(X_train.head())

# print('Train Set', X_train.info())
# print('Test Set', X_test.info())

# # TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



print(f"Train samples: {X_train_tfidf.shape[0]}")
print(f"Test samples: {X_test_tfidf.shape[0]}")

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMultinomial Naive bayes Classification Report:\n", classification_report(y_test, y_pred))

# Train Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)  # more iterations for convergence
logreg_model.fit(X_train_tfidf, y_train)

# Predict on the test data
y_pred_logreg = logreg_model.predict(X_test_tfidf)



def predict_spam(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(text_vector)[0]
    spam_score = proba[1] * 100  # spam class probability
    prediction = "Spam" if spam_score >= 50 else "Ham"
    return prediction, round(spam_score, 2)

sample_email = "Congratulations! You won a $1,000 Walmart gift card. Click to claim."
prediction, probability = predict_spam(sample_email, model, vectorizer)
print(sample_email)
print(f"Prediction: {prediction}")
print(f"Spam Probability: {probability}%")


# Evaluate Logistic Regression
print("=== Logistic Regression Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_logreg))

def predict_spam_logreg(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(text_vector)[0]
    spam_score = proba[1] * 100
    prediction = "Spam" if spam_score >= 50 else "Ham"
    return prediction, round(spam_score, 2)

sample_email = "Your account has been suspended! Please update your information."
print(sample_email)
prediction, probability = predict_spam_logreg(sample_email, logreg_model, vectorizer)

print(f"Prediction: {prediction}")
print(f"Spam Probability: {probability}%")


# Save models and vectorizer
joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(logreg_model, 'logreg_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

