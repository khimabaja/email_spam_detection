import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

# Download stopwords if not already present
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load CSV file
CSV_PATH = 'datasets/enron_spam_data.csv'  # update with your file path
df = pd.read_csv(CSV_PATH)

# print(df.head())
# print(df.info())

# Normalize labels if needed (e.g., "spam"/"ham" → 1/0)
df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})  # only if labels are strings
print(df.head())
# print(df.info())

# # Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]
    return ' '.join(tokens)

# # Apply preprocessing
df['clean_text'] = df['Message'].apply(preprocess_text)

print(df.head())

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

# print(X_test_tfidf)

print(f"Train samples: {X_train_tfidf.shape[0]}")
print(f"Test samples: {X_test_tfidf.shape[0]}")

# Hyperparameter Tuning for Naive Bayes
param_grid_nb = {'alpha': [0.1, 0.5, 1, 2, 5]}
grid_search_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=5, n_jobs=-1)
grid_search_nb.fit(X_train_tfidf, y_train)
print(f"Best params for Naive Bayes: {grid_search_nb.best_params_}")

# Train the best Naive Bayes model
best_nb_model = grid_search_nb.best_estimator_

# Predict on the test data
y_pred_nb = best_nb_model.predict(X_test_tfidf)

# Evaluate Naive Bayes Model
print("=== Naive Bayes Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

# Hyperparameter Tuning for Logistic Regression
param_grid_logreg = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
grid_search_logreg = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_logreg, cv=5, n_jobs=-1)
grid_search_logreg.fit(X_train_tfidf, y_train)
print(f"Best params for Logistic Regression: {grid_search_logreg.best_params_}")

# Train the best Logistic Regression model
best_logreg_model = grid_search_logreg.best_estimator_

# Predict on the test data
y_pred_logreg = best_logreg_model.predict(X_test_tfidf)

# Evaluate Logistic Regression Model
print("=== Logistic Regression Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_logreg))

# Calculate ROC-AUC for Naive Bayes
y_prob_nb = best_nb_model.predict_proba(X_test_tfidf)[:, 1]
roc_auc_nb = roc_auc_score(y_test, y_prob_nb)
print(f"ROC-AUC for Naive Bayes: {roc_auc_nb}")

# Calculate ROC-AUC for Logistic Regression
y_prob_logreg = best_logreg_model.predict_proba(X_test_tfidf)[:, 1]
roc_auc_logreg = roc_auc_score(y_test, y_prob_logreg)
print(f"ROC-AUC for Logistic Regression: {roc_auc_logreg}")

# Plot ROC curve for Naive Bayes
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})')

# Plot ROC curve for Logistic Regression
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {roc_auc_logreg:.2f})')

# Plot settings
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

def predict_spam(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(text_vector)[0]
    spam_score = proba[1] * 100  # spam class probability
    prediction = "Spam" if spam_score >= 50 else "Ham"
    return prediction, round(spam_score, 2)

sample_email = "Congratulations! You won a $1,000 Walmart gift card. Click to claim."
prediction, probability = predict_spam(sample_email, best_nb_model, vectorizer)
print(sample_email)
print(f"Prediction: {prediction}")
print(f"Spam Probability: {probability}%")

def predict_spam_logreg(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(text_vector)[0]
    spam_score = proba[1] * 100
    prediction = "Spam" if spam_score >= 50 else "Ham"
    return prediction, round(spam_score, 2)

sample_email = "Your account has been suspended! Please update your information."
print(sample_email)
prediction, probability = predict_spam_logreg(sample_email, best_logreg_model, vectorizer)

print(f"Prediction: {prediction}")
print(f"Spam Probability: {probability}%")

# Save the models and vectorizer
joblib.dump(best_nb_model, 'naive_bayes_model.pkl')
joblib.dump(best_logreg_model, 'logreg_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

