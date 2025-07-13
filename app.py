import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="💳 Fraud Detection App", layout="wide")

# -- HEADER UI --
st.markdown("""
    <style>
    .main-container {
        background: linear-gradient(to right, #e0f7fa, #e1f5fe);
        padding: 20px;
        border-radius: 15px;
    }
    .header {
        font-size: 40px;
        font-weight: 800;
        color: #1e88e5;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 20px;
        font-weight: 600;
        color: #0d47a1;
        background-color: #ffffffcc;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">🚨 Credit Card Fraud Detection System</div>', unsafe_allow_html=True)
st.markdown('<p class="subheader">📤 Upload your CSV transaction file and detect fraudulent activities in real-time.</p>', unsafe_allow_html=True)

# -- FILE UPLOAD --
uploaded_file = st.file_uploader("📤 Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Class' not in df.columns:
        st.error("❌ 'Class' column (label) not found in dataset.")
    else:
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head(10), use_container_width=True)

        # -- DATA BALANCING --
        legal = df[df['Class'] == 0]
        fraud = df[df['Class'] == 1]
        sample_legal = legal.sample(n=len(fraud), random_state=42)
        balanced_df = pd.concat([sample_legal, fraud])

        X = balanced_df.drop('Class', axis=1)
        y = balanced_df['Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # -- MODEL TRAINING --
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.markdown("### 📈 Model Performance:")
        st.text(classification_report(y_test, y_pred))

        st.markdown("### 🔍 Predict on All Uploaded Data:")
        pred_all = clf.predict(df.drop("Class", axis=1))
        df["Prediction"] = pred_all
        frauds_detected = df[df["Prediction"] == 1]

        st.metric("🚨 Predicted Fraud Transactions", len(frauds_detected))
        st.write(frauds_detected.head(10))

        st.download_button("⬇ Download Fraud Results as CSV", frauds_detected.to_csv(index=False), file_name="fraud_predictions.csv")
