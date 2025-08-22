import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# App Configuration
# -----------------------------------
st.set_page_config(layout="wide")

# Load saved ML model pipeline
with open('best_model.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# Extract regressor from pipeline (fallback to last step if not explicitly named)
try:
    regressor = model_pipeline.named_steps['regressor']
except KeyError:
    regressor = list(model_pipeline.named_steps.values())[-1]

# Load feature importance data from Excel
@st.cache_data
def load_feature_importance(file_path):
    return pd.read_excel(file_path)

final_fi = load_feature_importance("feature_importance.xlsx")

# -----------------------------------
# Sidebar: Mother Features Input
# -----------------------------------
image_sidebar = Image.open('Pic1.png')  # Sidebar image
st.sidebar.image(image_sidebar, use_container_width=True)
st.sidebar.header('Mother Features')

with st.sidebar.form("input_form"):
    mom_hs = st.number_input(
        'Mother attended high school?',
        min_value=0, max_value=1, value=0,
        help="Enter 1 if the mother attended high school, otherwise 0."
    )

    mom_iq = st.number_input(
        "Mother's IQ",
        min_value=50, max_value=160, value=100,
        help="Estimated IQ score of the mother (between 50 and 160)."
    )

    mom_work = st.number_input(
        "Mother's Work",
        min_value=1, max_value=4, value=1,
        help="Mother's work status (1=Unemployed, 2=Part-time, 3=Full-time, 4=Other)."
    )

    ppvt = st.number_input(
        "PPVT Score",
        min_value=0, max_value=200, value=100,
        help="Peabody Picture Vocabulary Test score, indicating verbal ability."
    )

    educ_cat = st.number_input(
        "Education Category",
        min_value=1, max_value=4, value=1,
        help="Education category (1=Primary, 2=Secondary, 3=College, 4=Postgraduate)."
    )

    mom_age_group = st.selectbox(
        "Mother Age Group",
        ['Teenager', 'Twenties'],
        help="Select the age group of the mother during childbirth."
    )

    submitted = st.form_submit_button("Predict", help="Click to predict the child's IQ based on inputs.")

# Store input data in dictionary
input_data = {
    'mom_hs': mom_hs,
    'mom_iq': mom_iq,
    'mom_work': mom_work,
    'ppvt': ppvt,
    'educ_cat': educ_cat,
    'mom_age_group': mom_age_group
}

# -----------------------------------
# Main Content: Banner & Title
# -----------------------------------
image_banner = Image.open('Pic2.png')  # Main header image
st.image(image_banner, use_container_width=True)
st.markdown("<h1 style='text-align: center;'>Child IQ Prediction App</h1>", unsafe_allow_html=True)

# -----------------------------------
# Layout: Feature Importance & Prediction
# -----------------------------------
col_main = st.container()
with col_main:
    col_top, col_bottom = st.columns([1, 1])

    # ---------------- Feature Importance ----------------
    with col_top:
        st.subheader("📊 Feature Importance", help="Shows how much each feature contributes to the prediction.")

        # Identify score column dynamically if needed
        if 'Feature Importance Score' in final_fi.columns:
            score_col = 'Feature Importance Score'
        else:
            score_col = final_fi.columns[1]  # fallback to second column

        # Sort and plot feature importance
        final_fi_sorted = final_fi.sort_values(score_col, ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=final_fi_sorted,
            x=score_col,
            y=final_fi_sorted.columns[0],
            palette='viridis',
            ax=ax
        )
        ax.set_title(f"Feature Importance - {type(regressor).__name__}")
        st.pyplot(fig)

    # ---------------- Prediction Section ----------------
    with col_bottom:
        st.subheader("🔮 Predict Child IQ", help="Generate the predicted IQ score based on the input values.")

        features = ['mom_hs', 'mom_iq', 'mom_work', 'ppvt', 'educ_cat', 'mom_age_group']

        def prepare_input(data, feature_list):
            """Prepare input data as a DataFrame for prediction."""
            return pd.DataFrame([{feature: data.get(feature, 0) for feature in feature_list}])

        # Run prediction if form submitted
        if submitted:
            input_df = prepare_input(input_data, features)
            prediction = model_pipeline.predict(input_df)
            st.success(f"🎓 Predicted Child IQ: **{prediction[0]:.2f}**", icon="✅")
