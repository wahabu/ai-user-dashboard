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
