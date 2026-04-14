import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Fraud AI System", layout="wide")

st.markdown("""
    <h1 style='text-align:center; color:#ff4b4b;'>💳 AI Fraud Detection System</h1>
""", unsafe_allow_html=True)

st.divider()

# ======================
# LOAD DATA (FAST - NO TRAINING)
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("dataset/creditcard.csv")
    df["Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))
    df = df.drop(["Time"], axis=1)
    return df

df = load_data()

X = df.drop("Class", axis=1)
y = df["Class"]

# ======================
# LOAD MODEL (FAST)
# ======================
model = joblib.load("fraud_model.pkl")

# predictions for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)

# ======================
# SIDEBAR MENU
# ======================
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Graphs", "Prediction"]
)

# ======================
# DASHBOARD
# ======================
if menu == "Dashboard":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Features", df.shape[1])

    with col3:
        st.metric("Accuracy", f"{accuracy:.4f}")

    st.success("System Running Fast ⚡")

# ======================
# GRAPHS
# ======================
elif menu == "Graphs":
    st.subheader("Fraud vs Normal")

    fig, ax = plt.subplots(figsize=(4, 3))
    df["Class"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    ax2.matshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig2)

    st.text(classification_report(y_test, y_pred))

# ======================
# PREDICTION
# ======================
elif menu == "Prediction":
    st.subheader("Test Transaction")

    input_data = []

    for col in X.columns[:5]:
        input_data.append(st.number_input(col, value=0.0))

    if st.button("Predict"):
        pred = model.predict([input_data])
        prob = model.predict_proba([input_data])[0][1] * 100

        if pred[0] == 1:
            st.error(f"⚠ Fraud Detected | Risk: {prob:.2f}%")
        else:
            st.success(f"Safe Transaction | Risk: {prob:.2f}%")