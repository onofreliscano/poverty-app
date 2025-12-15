import streamlit as st
import pandas as pd
import joblib
import gcsfs
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from models_utils import build_full_feature_df

# ------------------------------
# Load XGBoost model
# ------------------------------
@st.cache_resource
def load_model():
    # comment: Load trained model from models folder
    return joblib.load("models/notebooks_generated_best_model.pkl")

model = load_model()

st.title("Poverty Probability Predictor")

# ------------------------------
# Input fields for prediction
# ------------------------------
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
    "num_shocks_last_year": int(num_shocks_last_year)
}

# ------------------------------
# Prediction button
# ------------------------------
if st.button("Predict"):
    df_ready = build_full_feature_df(user_inputs)

    st.write("### DF Ready")
    st.write(df_ready)

    try:
        pred = model.predict(df_ready)[0]
        st.success(f"Poverty probability: {pred:.4f}")
    except Exception as e:
        st.error(f"Error: {e}")


# ====================================================
#   DATA VISUALIZATION (READ FROM GCS BUCKET)
# ====================================================

st.header("📊 Data Explorer (from Google Cloud Storage)")

@st.cache_resource
def load_data_from_gcs():
    """Load the training datasets directly from a GCS bucket."""
    fs = gcsfs.GCSFileSystem()

    base = "gs://4geeks-ds-lab-data/predicting-poverty"

    with fs.open(f"{base}/train_values_wJZrCmI.csv") as f:
        train_values = pd.read_csv(f)

    with fs.open(f"{base}/train_labels.csv") as f:
        train_labels = pd.read_csv(f)

    df = train_values.merge(train_labels, on="row_id", how="left")
    return df


if st.checkbox("Load dataset and show charts"):
    df = load_data_from_gcs()
    st.success("Data loaded from Google Cloud Storage!")

    st.write("### Sample of the dataset")
    st.write(df.head())

    # --------------------------
    # Histogram of poverty
    # --------------------------
    st.subheader("Histogram – Poverty Probability")
    fig, ax = plt.subplots()
    sns.histplot(df["poverty_probability"], kde=True, ax=ax)
    st.pyplot(fig)

    # --------------------------
    # Choropleth map
    # --------------------------
    COUNTRY_MAP = {
        "A": "KEN",
        "C": "TZA",
        "D": "UGA",
        "F": "ZMB",
        "G": "NGA",
        "I": "MWI",
        "J": "ETH",
    }

    df["iso"] = df["country"].map(COUNTRY_MAP)
    country_mean = df.groupby("iso")["poverty_probability"].mean().reset_index()

    st.subheader("Map – Average Poverty Probability per Country")
    fig_map = px.choropleth(
        country_mean,
        locations="iso",
        color="poverty_probability",
        color_continuous_scale="Reds",
        title="Average Poverty Probability"
    )
    st.plotly_chart(fig_map)

