import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load user data from CSV
df = pd.read_csv("users.csv")

# App title
st.title("📊 User Analytics Dashboard")

# Display raw data
st.subheader("📄 Raw Data Preview")
st.dataframe(df)

# Visualize number of users by country
st.subheader("🌍 User Count by Country")
fig, ax = plt.subplots()
sns.countplot(data=df, x="country", ax=ax)
st.pyplot(fig)


# ---------------------------
# 🧠 AI Prediction Section
# ---------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Create a binary label: 1 if purchases > 5, else 0
df['is_active_buyer'] = df['purchases'].apply(lambda x: 1 if x > 5 else 0)

# Features and labels
X = df[['age', 'last_login_days']]
y = df['is_active_buyer']

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Display classification report
st.subheader("🧠 AI Prediction Performance")
st.text(classification_report(y_test, y_pred))