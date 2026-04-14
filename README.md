# 💳 AI Fraud Detection System

An end-to-end Machine Learning based Fraud Detection System that identifies fraudulent financial transactions using real-world credit card transaction data. The system uses classification algorithms to analyze patterns and detect anomalies with high accuracy.

---

## 🚀 Project Overview

This project focuses on detecting fraudulent transactions using Machine Learning techniques. The model is trained on historical transaction data and can predict whether a transaction is normal or fraudulent.

A complete web-based dashboard is built using Streamlit to provide interactive visualization, real-time prediction, and model evaluation.

---

## ⚙️ Technologies Used

- Python 🐍  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  
- Joblib  

---

## 📊 Dataset Information

- Dataset Name: Credit Card Fraud Detection Dataset  
- Source: Kaggle  
- Link: https://www.kaggle.com/mlg-ulb/creditcardfraud  

This dataset contains anonymized credit card transactions with labeled fraud and non-fraud cases.

---

## 🧠 Machine Learning Workflow

The project follows a complete ML pipeline:

✔ Data Collection (Kaggle dataset)  
✔ Data Preprocessing  
- Feature scaling using StandardScaler  
- Removing irrelevant features (Time column)  

✔ Model Training  
- Algorithm used: Random Forest Classifier  
- Train-Test Split: 80/20  
- Handles class imbalance effectively  

✔ Model Evaluation  
- Accuracy scoring  
- Confusion Matrix  
- Classification Report  

✔ Model Saving  
- Trained model saved using Joblib for fast inference  

---

## 📈 Features of the System

✔ Fraud vs Normal transaction visualization  
✔ Real-time fraud prediction system  
✔ Fraud probability score (%) output  
✔ Confusion matrix visualization  
✔ Classification report display  
✔ Interactive Streamlit dashboard  
✔ Sidebar navigation (Dashboard / Graphs / Prediction)  
✔ Fast model loading (optimized performance)  

---

## 🖥️ Project Structure

AI_Fraud_Detection/
│
├── app.py                # Streamlit UI application  
├── train_once.py        # Model training script  
├── fraud_model.pkl      # Saved trained model  
├── README.md            # Project documentation  
└── dataset/             # Not uploaded due to size  

---

## 🚀 How to Run Project

### Step 1: Install dependencies
pip install pandas numpy scikit-learn matplotlib streamlit joblib

### Step 2: Train model (only first time)
python train_once.py

### Step 3: Run Streamlit app
streamlit run app.py

---

## 📊 Model Performance

- Accuracy: ~99%
- High precision in fraud detection
- Evaluated using confusion matrix and classification report
- Optimized for fast prediction

---

## 🎯 Key Highlights

- Real-world financial fraud detection use case  
- End-to-end Machine Learning pipeline  
- Interactive Streamlit dashboard  
- Real-time prediction system  
- Clean and optimized model architecture  
- Internship-ready professional project  

---

## 👩‍💻 Author

**Deepika Gautam**  
B.Tech (Artificial Intelligence & Machine Learning)  
Skills: Python, Machine Learning, Data Science, Streamlit  

---

## 📌 Future Improvements

- Integration with real-time API system  
- Advanced ML models (XGBoost / LightGBM)  
- Cloud deployment (AWS / Render)  
- Enhanced UI/UX improvements  
- Mobile responsive dashboard  

---

## ⭐ Acknowledgement

Special thanks to Kaggle for providing the dataset and open-source ML community for tools and libraries.

---

## ⭐ Support

If you like this project, don't forget to star ⭐ the repository!
