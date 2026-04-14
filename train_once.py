import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Training model once...")

df = pd.read_csv("dataset/creditcard.csv")

df["Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))
df = df.drop(["Time"], axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
model.fit(X_train, y_train)

joblib.dump(model, "fraud_model.pkl")

print("Model saved successfully ✔")