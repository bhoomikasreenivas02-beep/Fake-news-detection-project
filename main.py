import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
DATASET_FILE = "fake_news.csv"

try:
    df = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"'{DATASET_FILE}' was not found. Put the dataset CSV in the same "
        "folder as main.py and run the program again."
    )

print("\nDataset shape:", df.shape)
print("Columns:", list(df.columns))
print("\nFirst 5 records:")
print(df.head())

# ---------------------------------------------------------
# 2. Data Preprocessing
# ---------------------------------------------------------
required_columns = {"text", "label"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required column(s): {', '.join(sorted(missing))}. "
        "The dataset should contain at least 'text' and 'label'."
    )

df = df.dropna(subset=["text", "label"]).drop_duplicates().copy()
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("\nLabel counts:")
print(df["label"].value_counts())

# Use title + text when title is available
if "title" in df.columns:
    df["news_text"] = (
        df["title"].fillna("").astype(str) + " " +
        df["text"].fillna("").astype(str)
    )
else:
    df["news_text"] = df["text"].astype(str)

# ---------------------------------------------------------
# 3. Text Cleaning using NLP
# ---------------------------------------------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["news_text"].apply(clean_text)

print("\nCleaned text sample:")
print(df[["news_text", "clean_text"]].head())

# ---------------------------------------------------------
# 4. Feature Extraction using TF-IDF
# ---------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["clean_text"])
y = df["label"]

print("\nTF-IDF feature matrix shape:", X.shape)

# ---------------------------------------------------------
# 5. Train/Test Split (80/20)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# 6. Model Selection and Training
# ---------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
}

results = {}
trained_models = {}

print("\nModel Results")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    trained_models[name] = model
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }

    print(f"\n{name}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

# ---------------------------------------------------------
# 7. Choose Best Model
# ---------------------------------------------------------
best_name = max(results, key=lambda name: results[name]["Accuracy"])
best_model = trained_models[best_name]

print("\nBest Model:", best_name)

# ---------------------------------------------------------
# 8. Detailed Evaluation
# ---------------------------------------------------------
best_predictions = best_model.predict(X_test)

print("\nClassification Report")
print("=" * 60)
print(classification_report(y_test, best_predictions, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. Model Accuracy Comparison
# ---------------------------------------------------------
model_names = list(results.keys())
accuracies = [results[name]["Accuracy"] for name in model_names]

plt.figure(figsize=(8, 5))
plt.bar(model_names, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. Prediction for New News
# ---------------------------------------------------------
print("\nEnter a news article to classify.")
new_news = input("News article: ").strip()

if new_news:
    cleaned_news = clean_text(new_news)
    new_news_tfidf = tfidf.transform([cleaned_news])
    prediction = best_model.predict(new_news_tfidf)[0]

    if prediction == 0:
        print("\nPrediction: Fake News")
    elif prediction == 1:
        print("\nPrediction: Real News")
    else:
        print(f"\nPrediction: Unknown label ({prediction})")
else:
    print("\nNo news article entered. Program finished.")
