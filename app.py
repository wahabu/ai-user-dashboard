import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------
# 📥 Load Data
# ---------------------------
df = pd.read_csv("users.csv")

# ---------------------------
# 🎨 Page Title
# ---------------------------
st.title("📊 User Analytics Dashboard")

# ---------------------------
# 🧾 Show Raw Data
# ---------------------------
st.subheader("📄 Raw Data Preview")
st.dataframe(df)

# ---------------------------
# 📊 Visual Analytics
# ---------------------------
st.subheader("🌍 User Count by Country")
fig, ax = plt.subplots()
sns.countplot(data=df, x="country", ax=ax)
st.pyplot(fig)

# ---------------------------
# 🧠 AI Model Setup
# ---------------------------
# Create a binary label: 1 if purchases > 5, else 0
df['is_active_buyer'] = df['purchases'].apply(lambda x: 1 if x > 5 else 0)

# Features and labels
X = df[['age', 'last_login_days']]
y = df['is_active_buyer']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# 🔍 Predict on User Input
# ---------------------------
st.subheader("🔍 Predict Purchase Intent for a New User")

age_input = st.number_input("Enter user's age", min_value=10, max_value=100, value=30)
login_days_input = st.number_input("Enter days since last login", min_value=0, max_value=365, value=10)

if st.button("Predict"):
    new_data = pd.DataFrame([[age_input, login_days_input]], columns=['age', 'last_login_days'])
    prediction = model.predict(new_data)[0]
    prob = model.predict_proba(new_data)[0][1]

    st.write(f"🔢 Purchase Probability: {prob:.2f}")

    if prediction == 1:
        st.success("✅ Likely to purchase again (active buyer)")
    else:
        st.warning("⚠️ Unlikely to purchase (inactive buyer)")

# ---------------------------
# 📈 Model Evaluation
# ---------------------------
st.subheader("🧠 AI Prediction Performance")

y_pred = model.predict(X_test)
st.text("Model Accuracy: {:.2f}%".format(model.score(X_test, y_test) * 100))
st.text(classification_report(y_test, y_pred))