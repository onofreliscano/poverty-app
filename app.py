import streamlit as st
import pandas as pd
import joblib
import gcsfs
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import shap

from models_utils import build_full_feature_df


# ====================================================
#   LOAD MODEL (FROM GCS)
# ====================================================

@st.cache_resource
def load_model():
    # comment: Load trained model from GCS bucket
    fs = gcsfs.GCSFileSystem()
    with fs.open(
        "gs://4geeks-ds-lab-data/notebooks/generated/best_model_avance2.pkl",
        "rb"
    ) as f:
        return joblib.load(f)


model = load_model()

st.title("Poverty Probability Predictor")
st.write("DEBUG – pipeline steps:")
st.write(model.named_steps)


# ====================================================
#   INPUT FIELDS
# ====================================================

country = st.selectbox("Country", ["C", "A", "D", "G", "F", "I", "J"])
age = st.number_input("Age", 0, 120, 25)
is_urban = st.selectbox("Urban?", ["Yes", "No"])
female = st.selectbox("Female?", ["Yes", "No"])
education_level = st.number_input("Education level", 0, 40, 10)
num_shocks_last_year = st.number_input("Num shocks last year", 0, 10, 0)

user_inputs = {
    "country": country,
    "age": int(age),
    "is_urban": 1 if is_urban == "Yes" else 0,
    "female": 1 if female == "Yes" else 0,
    "education_level": int(education_level),
    "num_shocks_last_year": int(num_shocks_last_year),
}


# ====================================================
#   PREDICTION
# ====================================================

if st.button("Predict v1.3"):
    df_ready = build_full_feature_df(user_inputs)
    st.session_state["df_ready"] = df_ready

    st.write("### DF Ready")
    st.write(df_ready)

    try:
        pred = model.predict(df_ready)[0]
        st.session_state["prediction"] = pred
        st.success(f"Poverty probability: {pred:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")


# ====================================================
#   SHAP VALUES (ON-DEMAND, SAFE)
# ====================================================

#@st.cache_resource
def get_shap_explainer(model):
    # comment: Create SHAP TreeExplainer without caching (safe)
    return shap.TreeExplainer(model)


if st.button("Show SHAP explanation"):
    if "df_ready" not in st.session_state:
        st.warning("Please run a prediction first.")
    else:
        df_ready = st.session_state["df_ready"]

        with st.spinner("Computing SHAP values..."):
            explainer = get_shap_explainer(model)
            shap_values = explainer.shap_values(df_ready)

            fig, ax = plt.subplots()
            shap.summary_plot(
                shap_values,
                df_ready,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig)


# ====================================================
#   DATA VISUALIZATION (FROM GCS)
# ====================================================

st.header("📊 Data Explorer (from Google Cloud Storage)")

@st.cache_resource
def load_data_from_gcs():
    # comment: Load training datasets directly from GCS
    fs = gcsfs.GCSFileSystem()
    base = "gs://4geeks-ds-lab-data/predicting-poverty"

    with fs.open(f"{base}/train_values_wJZrCmI.csv") as f:
        train_values = pd.read_csv(f)

    with fs.open(f"{base}/train_labels.csv") as f:
        train_labels = pd.read_csv(f)

    return train_values.merge(train_labels, on="row_id", how="left")


if st.checkbox("Load dataset and show charts"):
    df = load_data_from_gcs()
    st.success("Data loaded from Google Cloud Storage!")

    st.write("### Sample of the dataset")
    st.write(df.head())

    # --------------------------
    # Histogram
    # --------------------------
    st.subheader("Histogram – Poverty Probability")
    fig, ax = plt.subplots()
    sns.histplot(df["poverty_probability"], kde=True, ax=ax)
    st.pyplot(fig)

    # --------------------------
    # Choropleth map
    # --------------------------
    COUNTRY_MAP = {
        "A": "MEX",  # Mexico
        "C": "COL",  # Colombia
        "D": "PER",  # Peru
        "F": "CHL",  # Chile
        "G": "ARG",  # Argentina
        "I": "BRA",  # Brazil
        "J": "VEN",  # Venezuela
    }

    df["iso"] = df["country"].map(COUNTRY_MAP)
    country_mean = df.groupby("iso")["poverty_probability"].mean().reset_index()

    st.subheader("Map – Average Poverty Probability per Country")
    fig_map = px.choropleth(
        country_mean,
        locations="iso",
        color="poverty_probability",
        color_continuous_scale="Reds",
        title="Average Poverty Probability",
    )
    st.plotly_chart(fig_map)
