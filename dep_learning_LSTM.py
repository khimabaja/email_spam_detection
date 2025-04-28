# lstm_model.py

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# NLTK downloads (only first time needed)
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('datasets/enron_spam_data.csv')

# Map labels
df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})

# Preprocessing (if already cleaned, you can skip or reapply)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\n|\r', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = [word for word in text.split() if word not in stop_words]
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]
    return ' '.join(tokens)

# If you don't have "clean_text" column from previous step
df['clean_text'] = df['Message'].apply(preprocess_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# Tokenizer
vocab_size = 5000
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
max_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Build LSTM Model
lstm_model = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
lstm_model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_test_pad, y_test))

# Evaluate
loss, accuracy = lstm_model.evaluate(X_test_pad, y_test)
print("\n=== LSTM Model Results ===")
print(f"Test Accuracy: {accuracy:.4f}")

# ROC-AUC
y_pred_prob = lstm_model.predict(X_test_pad).ravel()
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"LSTM ROC AUC Score: {auc_score:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label="LSTM")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for LSTM")
plt.legend()
plt.grid()
plt.show()

# Optionally save the model
lstm_model.save('lstm_spam_classifier.h5')
