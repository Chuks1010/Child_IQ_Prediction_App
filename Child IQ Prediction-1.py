import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Config
st.set_page_config(layout="wide")

# Load model pipeline
with open('best_model.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# Try to get model name
try:
    regressor = model_pipeline.named_steps['regressor']
except KeyError:
    regressor = list(model_pipeline.named_steps.values())[-1]

# Load feature importance
@st.cache_data
def load_feature_importance(file_path):
    return pd.read_excel(file_path)

final_fi = load_feature_importance("feature_importance.xlsx")

# ----------------------------
# Sidebar (Mother and Child Input)
# ----------------------------

image_sidebar = Image.open('Pic1.png')  # Replace with your image
st.sidebar.image(image_sidebar, use_container_width=True)
st.sidebar.header('Mother Features')

with st.sidebar.form("input_form"):
    mom_hs = st.number_input('Mother\'s High School Score', min_value=0, max_value=1, value=0)
    mom_iq = st.number_input('Mother\'s IQ', min_value=50, max_value=160, value=100)
    mom_work = st.number_input('Mother\'s Work Hours/Week', min_value=1, max_value=4, value=1)
    ppvt = st.number_input('PPVT Score', min_value=0, max_value=200, value=100)
    educ_cat = st.number_input('Education Category', min_value=1, max_value=4, value=1)
    mom_age_group = st.selectbox('Mother Age Group', ['Teenager', 'Twenties'])
    submitted = st.form_submit_button("Predict")

input_data = {
    'mom_hs': mom_hs,
    'mom_iq': mom_iq,
    'mom_work': mom_work,
    'ppvt': ppvt,
    'educ_cat': educ_cat,
    'mom_age_group': mom_age_group
}

# ----------------------------
# Main App Content
# ----------------------------

image_banner = Image.open('Pic2.png')  # Replace with your image
st.image(image_banner, use_container_width=True)
st.markdown("<h1 style='text-align: center;'>Child IQ Prediction App</h1>", unsafe_allow_html=True)



# Right Column (Split vertically)
col_main = st.container()
with col_main:
    col_top, col_bottom = st.columns([1, 1])

    # ---- Top Right: Feature Importance
    with col_top:
        st.subheader("📊 Feature Importance")
        
        # Debug: show column names if error persists
        # st.dataframe(final_fi)
        # st.write("Columns:", final_fi.columns.tolist())
        
        # Replace below with actual column names in your Excel file
        if 'Feature Importance Score' in final_fi.columns:
            score_col = 'Feature Importance Score'
        else:
            score_col = final_fi.columns[1]  # fallback: second column
        
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

    # ---- Bottom Right: Prediction
    with col_bottom:
        st.subheader("🔮 Predict Child IQ")

        features = ['mom_hs', 'mom_iq', 'mom_work', 'ppvt', 'educ_cat', 'mom_age_group']

        def prepare_input(data, feature_list):
            return pd.DataFrame([{feature: data.get(feature, 0) for feature in feature_list}])

        if submitted:
            input_df = prepare_input(input_data, features)
            prediction = model_pipeline.predict(input_df)
            st.success(f"🎓 Predicted Child IQ: **{prediction[0]:.2f}**")

# streamlit run 'Child IQ Prediction-1.py'