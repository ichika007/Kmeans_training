import streamlit as st

st.title("🎈 My WINE AI ")
st.write(
    "Sweet or sour"
)
import streamlit as st
import pandas as pd
import pickle

# ============================================================
# Load Model + Features
# ============================================================
with open("kmeans_wine_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("wine_features.pkl", "rb") as f:
    features = pickle.load(f)

# ============================================================
# Cluster Descriptions (EDIT THESE AS YOU LIKE)
# ============================================================
cluster_descriptions = {
    0: "Cluster 0 wines are typically lighter and may have higher acidity. They can be associated with younger, fresher profiles.",
    1: "Cluster 1 wines tend to be more balanced with medium acidity and body. These wines often fall into the 'average quality' range.",
    2: "Cluster 2 wines are usually richer, lower in acidity, and may indicate stronger flavor intensity or higher quality."
}

# ============================================================
# UI Header
# ============================================================
st.title("🍷 Wine Clustering App (K-Means)")
st.write("""
Upload **winequality-red.csv** so the app can analyze clusters  
and show similar wines from the dataset.
""")

# ============================================================
# Upload CSV File
# ============================================================
uploaded_file = st.file_uploader("Upload winequality-red.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    df_numeric = df[features]

    st.success("CSV loaded successfully!")

    # ============================================================
    # Input Form
    # ============================================================
    st.header("🔧 Enter New Wine Data")

    inputs = {}
    for col in features:
        default_value = float(df[col].mean())
        inputs[col] = st.number_input(
            f"{col}", value=default_value, format="%.4f"
        )

    new_data = pd.DataFrame([inputs], columns=features)

    # ============================================================
    # Predict Cluster
    # ============================================================
    if st.button("🔍 Predict Cluster"):
        cluster = model.predict(new_data)[0]

        st.success(f"📌 This wine belongs to **Cluster {cluster}**")

        # Show description
        st.info(f"📝 **Cluster {cluster} Description:**\n\n{cluster_descriptions.get(cluster, 'No description available.')}")

        # Assign cluster labels to dataset
        df_numeric["cluster"] = model.predict(df_numeric)

        # Show similar wines
        similar = df_numeric[df_numeric["cluster"] == cluster]

        st.subheader("📊 Similar Wines From This Cluster")
        st.write(similar.head(10))

else:
    st.warning("Please upload winequality-red.csv to continue.")


