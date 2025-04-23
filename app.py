import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- Step 1: Load the Dataset ---
@st.cache
def load_data():
    # Replace with the correct path to your dataset if needed
    df = pd.read_csv("dataset_traffic_accident_prediction1.csv")
    return df

# --- Step 2: Data Cleaning ---
def clean_data(df):
    # Fill missing numeric columns with median
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical columns with mode
    categorical_cols = df.select_dtypes(include="object").columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encoding categorical features
    le = LabelEncoder()
    for col in categorical_cols:
        if col != "Accident":  # We don't want to encode the target column
            df[col] = le.fit_transform(df[col])

    return df

# --- Step 3: Train the Model ---
def train_model(df):
    X = df.drop("Accident", axis=1)
    y = df["Accident"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc

# --- Step 4: Make Predictions ---
def make_prediction(model, input_data, columns):
    input_df = pd.DataFrame([input_data], columns=columns)
    prediction = model.predict(input_df)[0]
    return prediction

# --- Streamlit UI ---
st.title("🚗 Accident Prediction App")

# Load and clean the data
df = load_data()
df = clean_data(df)

# Train the model
model, acc = train_model(df)
st.success(f"Model trained! ✅ Accuracy: {acc:.2f}")

# Input for new prediction
st.subheader("Make a Prediction")

input_data = {}

# User inputs for each feature
for col in df.drop("Accident", axis=1).columns:
    if df[col].dtype == np.number:
        input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].median()))
    else:
        options = list(df[col].unique())
        input_data[col] = st.selectbox(f"Select {col}", options)

# Predict button
if st.button("Predict Accident"):
    prediction = make_prediction(model, input_data, df.drop("Accident", axis=1).columns)
    if prediction == 1:
        st.error("⚠️ Accident Predicted!")
    else:
        st.success("✅ No Accident Predicted!")
