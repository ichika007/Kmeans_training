import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# ============================================================
# Load Model + Features
# ============================================================
with open("kmeans_wine_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("wine_features.pkl", "rb") as f:
    features = pickle.load(f)

# Load dataset for showing similar samples
df = pd.read_csv("winequality-red.csv", sep=';')
df_numeric = df[features]

st.title("🍷 Wine Clustering App (K-Means)")

st.write("""
This app predicts which **cluster/group** a new wine belongs to  
based on the trained **K-Means (k=3)** model.
""")

# ============================================================
# Create input form for new data
# ============================================================
st.header("🔧 Enter New Wine Data")

inputs = {}
for col in features:
    default_value = float(df[col].mean())
    inputs[col] = st.number_input(
        f"{col}",
        value=default_value,
        format="%.4f"
    )

# Convert input to DataFrame
new_data = pd.DataFrame([inputs], columns=features)

# ============================================================
# Predict cluster
# ============================================================
if st.button("🔍 Predict Cluster"):
    cluster = model.predict(new_data)[0]
    st.success(f"📌 This wine belongs to **Cluster {cluster}**")

    # Show similar samples
    st.subheader("📊 Similar Wines (Same Cluster)")
    
    # Assign cluster labels for existing data
    df_numeric["cluster"] = model.predict(df_numeric)

    similar_wines = df_numeric[df_numeric["cluster"] == cluster]

    st.write(f"Found **{len(similar_wines)}** similar wines from the dataset.")

    st.dataframe(similar_wines.head(10))

    st.info("Showing first 10 similar wines from this cluster.")


