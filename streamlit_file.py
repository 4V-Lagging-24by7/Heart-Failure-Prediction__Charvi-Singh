import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description
st.title("Heart Failure Prediction")
st.write("Upload a dataset and predict heart failure using trained machine learning models.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())
    
    # Check for missing values
    st.write("Checking for missing values...")
    if data.isnull().sum().any():
        st.write("Missing values detected. Filling missing values with mean (for numerical columns) and mode (for categorical columns)...")
        # Fill missing values for numerical columns with mean
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col].fillna(data[col].mean(), inplace=True)
        # Fill missing values for categorical columns with mode
        for col in data.select_dtypes(include=['object']).columns:
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Data preprocessing
    st.write("Preprocessing the data...")
    target = st.selectbox("Select the target column (dependent variable):", data.columns)
    features = st.multiselect("Select feature columns (independent variables):", data.columns, default=data.columns[:-1])
    
    X = data[features]
    y = data[target]
    
    # Encode categorical variables
    st.write("Encoding categorical variables...")
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    
    if y.dtypes == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    st.write("Training a Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    # Feature importance
    st.write("Feature Importance:")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
    st.bar_chart(feature_importance_df.set_index("Feature"))
    
    # Visualization
    st.write("Correlation Heatmap:")
    # Filter only numeric columns for correlation
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    
    # Plot heatmap
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(plt)

else:
    st.write("Please upload a CSV file to proceed.")
