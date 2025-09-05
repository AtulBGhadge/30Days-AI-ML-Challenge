import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="🏙️ Dubai House Price Predictor",
    layout="wide",
    page_icon="🏠"
)

# ================================
# LOAD MODEL
# ================================
@st.cache_resource
def load_model():
    return joblib.load("rf_with_title_tfidf.joblib")

model = load_model()

# ================================
# HEADER
# ================================
st.title("🏠 Dubai House Price Prediction App")
st.markdown("""
This interactive app predicts **house prices in Dubai** using **Machine Learning**.  
Explore predictions, feature importance, and model insights.
""")

# ================================
# SIDEBAR - USER INPUT FORM
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

# ================================
# PREDICTION
# ================================
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

if st.sidebar.button("🔮 Predict Price"):
    prediction = model.predict(input_data)[0]
    prediction_price = np.expm1(prediction)

    st.subheader("💰 Prediction Result")
    st.success(f"Estimated Property Price: **AED {prediction_price:,.0f}**")

# ================================
# TABS FOR ANALYSIS
# ================================
tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "🔍 SHAP Explanation", "📈 Market Insights"])

# ---------- TAB 1: Feature Importance ----------
with tab1:
    st.subheader("Top Feature Importance (Permutation)")
    try:
        from sklearn.inspection import permutation_importance

        # Use small sample for speed
        X_sample = input_data.copy()
        r = permutation_importance(model, X_sample, model.predict(X_sample), n_repeats=3, random_state=42)

        feat_imp = pd.Series(r.importances_mean, index=X_sample.columns).sort_values(ascending=False)
        plt.figure(figsize=(8,4))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis")
        plt.title("Top Features Driving Predictions")
        st.pyplot(plt)
    except Exception as e:
        st.warning(f"Feature importance not available: {e}")

# ---------- TAB 2: SHAP Explanation ----------
with tab2:
    st.subheader("Explain Prediction with SHAP")
    try:
        explainer = shap.TreeExplainer(model.named_steps['regressor'])
        X_transformed = model.named_steps['preprocessor'].transform(input_data)
        shap_values = explainer.shap_values(X_transformed)

        st.write("SHAP Summary Plot:")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_transformed, feature_names=input_data.columns, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")

# ---------- TAB 3: Market Insights ----------
with tab3:
    st.subheader("Dubai Market Insights")
    st.markdown("Here you can add dataset visualizations (histograms, correlations, etc.)")

    # Example distribution (replace with your dataset)
    fake_prices = np.random.lognormal(mean=14, sigma=0.5, size=5000)
    plt.figure(figsize=(8,4))
    sns.histplot(fake_prices, bins=50, kde=True)
    plt.xlabel("Price (AED)")
    plt.title("Distribution of House Prices in Dubai")
    st.pyplot(plt)
