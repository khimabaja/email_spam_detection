import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load saved models
nb_model = joblib.load('naive_bayes_model.pkl')
logreg_model = joblib.load('logreg_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Download stopwords if not already done
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function (same as training)
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Prediction function
def predict_spam(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    proba = model.predict_proba(text_vector)[0]
    spam_score = proba[1] * 100
    prediction = "Spam" if spam_score >= 50 else "Ham"
    return prediction, round(spam_score, 2)

# Streamlit App
st.title("📧 Spam vs Ham Classifier")

# Sidebar to choose model
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ("Multinomial Naive Bayes", "Logistic Regression")
)

# Input text
user_input = st.text_area("Enter the email or message here:", height=200)

# Predict button
if st.button("Predict"):
    if user_input.strip() != "":
        if model_choice == "Multinomial Naive Bayes":
            selected_model = nb_model
        else:
            selected_model = logreg_model

        prediction, spam_percentage = predict_spam(user_input, selected_model, vectorizer)
        
        st.markdown(f"### 📢 Prediction: **{prediction}**")
        st.markdown(f"### 📊 Spam Probability: **{spam_percentage}%**")
    else:
        st.warning("Please enter a message to classify.")
