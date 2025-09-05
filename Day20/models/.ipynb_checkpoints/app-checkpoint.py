import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ================================
# 🔹 Custom Transformer for Model
# ================================
class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.column]

# ================================
# 🔹 PAGE CONFIG
# ================================
st.set_page_config(
    page_title="🏙️ Dubai House Price Predictor",
    layout="wide",
    page_icon="🏠"
)

# ================================
# 🔹 LOAD MODEL
# ================================
import os

@st.cache_resource
def load_model():
    model = load_model()
    model_path = os.path.join(os.path.dirname(__file__), "rf_with_title_tfidf.joblib")
    return joblib.load(model_path)


# ================================
# 🔹 HEADER
# ================================
st.title("🏠 Dubai House Price Prediction App")
st.markdown("""
Welcome to the **Dubai Property Price Predictor** 🚀  

This app uses **Machine Learning** to estimate property prices in Dubai.  
Provide your property details → get predictions, market insights, and model explanations.
""")

# ================================
# 🔧 SIDEBAR - USER INPUT
# ================================
st.sidebar.header("🔧 Input Features")

bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 10, 2)
size_min_clean = st.sidebar.number_input("Area (sqft)", 200, 20000, 1200, step=50)

title = st.sidebar.text_input("Property Title", "Luxury apartment in Dubai Marina")
title_len = len(title.split())
desc_len = int(size_min_clean / 50)  # proxy

location = st.sidebar.selectbox("Location", ["Dubai Marina", "Downtown", "JVC", "Palm Jumeirah", "Business Bay"])
ptype = st.sidebar.selectbox("Property Type", ["Apartment", "Villa", "Townhouse", "Penthouse"])
furnishing = st.sidebar.selectbox("Furnishing", ["Furnished", "Unfurnished", "Semi-Furnished"])
verified = st.sidebar.radio("Verified Listing?", ["Yes", "No"])
priceDuration = st.sidebar.radio("Price Duration", ["Monthly", "Yearly"])

# Build input dataframe
input_data = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "size_min_clean": size_min_clean,
    "title": title,
    "title_len": title_len,
    "desc_len": desc_len,
    "displayAddress_top": location,
    "type": ptype,
    "furnishing": furnishing,
    "verified": verified,
    "priceDuration": priceDuration,
    "price_per_sqft": size_min_clean / max(1, bedrooms + bathrooms)
}])

# ================================
# 🔮 PREDICTION
# ================================
if st.sidebar.button("🔮 Predict Price"):
    prediction = model.predict(input_data)[0]
    prediction_price = np.expm1(prediction)

    st.success(f"💰 **Estimated Price: AED {prediction_price:,.0f}**")

    # 📌 Property Summary Card
    with st.container():
        st.subheader("🏠 Property Overview")
        st.write(f"- **Bedrooms:** {bedrooms}")
        st.write(f"- **Bathrooms:** {bathrooms}")
        st.write(f"- **Size:** {size_min_clean} sqft")
        st.write(f"- **Type:** {ptype}")
        st.write(f"- **Furnishing:** {furnishing}")
        st.write(f"- **Verified:** {verified}")
        st.write(f"- **Location:** {location}")

    # 📌 Smart Recommendation
    if prediction_price > 2_000_000:
        st.info("💡 Premium property detected. Great for **luxury buyers**!")
    elif prediction_price > 800_000:
        st.success("✅ Mid-range property. Good **investment opportunity**.")
    else:
        st.warning("📉 Affordable property. Check for **hidden opportunities**.")

# ================================
# 📊 TABS
# ================================
tab1, tab2, tab3 = st.tabs(["📈 Market Insights", "📊 Feature Importance", "🔍 Model Explanation"])

# ---------- TAB 1: Market Insights ----------
with tab1:
    st.subheader("Dubai Market Insights")

    fake_data = pd.DataFrame({
        "Location": np.random.choice(["Dubai Marina", "Downtown", "Palm Jumeirah"], 200),
        "Price": np.random.lognormal(mean=14, sigma=0.4, size=200)
    })
    avg_prices = fake_data.groupby("Location")["Price"].mean()

    st.bar_chart(avg_prices)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(fake_data["Price"], bins=40, kde=True, ax=ax)
    ax.set_xlabel("Price (AED)")
    ax.set_title("Distribution of House Prices")
    st.pyplot(fig)

# ---------- TAB 2: Feature Importance ----------
with tab2:
    st.subheader("Top Features (from RandomForest)")
    try:
        importances = model.named_steps["regressor"].feature_importances_
        feature_names = input_data.columns
        feat_imp = pd.Series(importances[:len(feature_names)], index=feature_names).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10], palette="viridis", ax=ax)
        ax.set_title("Top 10 Features Driving Predictions")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Feature importance not available: {e}")

# ---------- TAB 3: SHAP Explanation ----------
with tab3:
    st.subheader("Explain Prediction with SHAP")
    try:
        explainer = shap.TreeExplainer(model.named_steps['regressor'])

        X_transformed = model.named_steps['preprocessor'].transform(input_data)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        X_transformed = np.array(X_transformed, dtype=np.float32)

        shap_values = explainer.shap_values(X_transformed)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed, show=False)  # ❌ removed feature_names
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")
