import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Title and Description
st.title("Car Price Prediction App")
st.write("This application allows you to explore car pricing data and predict prices using a linear regression model.")

# Load Data
@st.cache
def load_data():
    # Replace 'car_data.csv' with the actual file path of your dataset
    df = pd.read_csv('car data.csv')
    return df

data = load_data()

# Show Raw Data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data.head())

# Data Overview
st.subheader("Dataset Overview")
st.write(f"Number of rows: {data.shape[0]}")
st.write(f"Number of columns: {data.shape[1]}")
st.write(data.describe())

# Check for Missing Values
st.subheader("Missing Values")
missing_values = data.isnull().sum()
st.write(missing_values)

# Handling Missing Values (if any)
data = data.dropna()  # Drop rows with missing values. You can also choose to fill missing values instead.

# Exploratory Data Analysis (EDA)
st.subheader("Exploratory Data Analysis")
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

# Correlation Matrix
st.write("Correlation Matrix")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Scatter Plot
st.write("Scatter Plot")
x_axis = st.selectbox("Choose X-axis", numeric_cols)
y_axis = st.selectbox("Choose Y-axis", numeric_cols)
fig, ax = plt.subplots()
ax.scatter(data[x_axis], data[y_axis])
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
st.pyplot(fig)

# Data Preprocessing for Model
st.subheader("Model Training")

# Encoding categorical variables
label_encoders = {}
categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Target and Features
target = st.selectbox("Select Target Variable", numeric_cols)
features = st.multiselect("Select Feature Variables", [col for col in numeric_cols if col != target])

if features:
    X = data[features]
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Display Metrics
    st.subheader("Model Performance")
    st.write("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
    st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    st.write("R-squared Score:", r2_score(y_test, y_pred))

    # Prediction vs Actual Plot
    st.subheader("Prediction vs Actual")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    st.pyplot(fig)
else:
    st.write("Please select the target and feature variables for training.")
