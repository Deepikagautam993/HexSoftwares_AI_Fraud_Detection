import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

print("💳 AI Fraud Detection System Started")

# ======================
# 1. LOAD DATASET
# ======================
df = pd.read_csv("dataset/creditcard.csv")

print("Dataset Shape:", df.shape)

# ======================
# 2. PREPROCESSING
# ======================
df["Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))
df = df.drop(["Time"], axis=1)

# ======================
# 3. FEATURES & TARGET
# ======================
X = df.drop("Class", axis=1)
y = df["Class"]

# ======================
# 4. TRAIN TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training model... ⏳")

# ======================
# 5. MODEL TRAINING
# ======================
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Model training completed ✔")

# ======================
# 6. PREDICTION
# ======================
y_pred = model.predict(X_test)

# ======================
# 7. RESULTS
# ======================
print("\n====================")
print("MODEL RESULTS")
print("====================")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ======================
# 8. GRAPH (POPUP WINDOW)
# ======================
print("Showing graph...")

plt.figure(figsize=(6,4))
df["Class"].value_counts().plot(kind="bar")

plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Count")

plt.show()

print("Graph displayed ✔")