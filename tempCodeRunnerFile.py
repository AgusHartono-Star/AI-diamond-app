import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Diamond Price AI",
    page_icon="💎",
    layout="wide"
)

# =========================
# LOAD CSS
# =========================

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# =========================
# LOAD MODEL
# =========================

model = joblib.load("model_diamonds_final.pkl")

# =========================
# HERO SECTION
# =========================

st.image("assets/diamond_banner.png", use_container_width=True)

st.markdown(
"""
<div class="hero-title">💎 Diamond Price AI</div>
<div class="hero-subtitle">
Predict diamond prices instantly using Machine Learning
</div>
""",
unsafe_allow_html=True
)

st.write("")
st.write("")

# =========================
# ABOUT SECTION
# =========================

st.markdown('<div class="section-title">About This AI</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.image("assets/ai.webp", use_container_width=True)

with col2:
    st.write("""
This application predicts **diamond prices** using Machine Learning models.

Dataset used:
- Diamonds dataset (ggplot2)

Models implemented:
- K-Nearest Neighbors
- Random Forest
- XGBoost

The AI analyzes diamond characteristics such as:

• Carat  
• Cut  
• Color  
• Clarity  
• Dimensions  

and estimates the market price of the diamond.
""")

st.write("")
st.write("---")

# =========================
# PREDICTION SECTION
# =========================

st.markdown('<div class="section-title">Diamond Price Prediction</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    st.subheader("Diamond Features")

    carat = st.slider("Carat",0.2,5.0,1.0)
    depth = st.slider("Depth",40.0,80.0,60.0)
    table = st.slider("Table",40.0,80.0,55.0)

    x = st.slider("Length (x)",0.0,10.0,5.0)
    y = st.slider("Width (y)",0.0,10.0,5.0)
    z = st.slider("Height (z)",0.0,10.0,3.0)

    cut = st.selectbox("Cut",['Fair','Good','Very Good','Premium','Ideal'])
    color = st.selectbox("Color",['J','I','H','G','F','E','D'])
    clarity = st.selectbox("Clarity",['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])

with col2:

    st.subheader("Prediction Result")

    input_data = pd.DataFrame({
        "carat":[carat],
        "depth":[depth],
        "table":[table],
        "x":[x],
        "y":[y],
        "z":[z],
        "cut":[cut],
        "color":[color],
        "clarity":[clarity]
    })

    if st.button("💎 Predict Price"):

        price = model.predict(input_data)[0]

        st.metric(
            label="Estimated Diamond Price",
            value=f"${price:,.2f}"
        )

st.write("")
st.write("---")

# =========================
# MODEL INSIGHTS
# =========================

st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)

try:

    model_reg = model.named_steps["reg"]

    importance = model_reg.feature_importances_

    features = [
        "carat","depth","table","x","y","z",
        "cut","color","clarity"
    ]

    fig, ax = plt.subplots()

    ax.barh(features, importance)
    ax.set_title("Feature Importance")

    st.pyplot(fig)

except:
    st.info("Feature importance available only for RandomForest or XGBoost models")

st.write("")
st.write("---")

# =========================
# FOOTER
# =========================

st.markdown(
"""
<center>

Diamond Price AI  
Machine Learning Project using Streamlit

</center>
""",
unsafe_allow_html=True
)