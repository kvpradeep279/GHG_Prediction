import streamlit as st
import pandas as pd
import joblib

# --- Load Model and Preprocessing Objects ---
model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("features.pkl")

substace_encoder = joblib.load("substance_encoder.pkl")
unit_encoder = joblib.load("unit_encoder.pkl")
source_encoder = joblib.load("source_encoder.pkl")

# --- Streamlit Page Settings ---
st.set_page_config(page_title="GHG Emission Predictor", layout="wide")
st.title("🌱 Greenhouse Gas (GHG) Emission Predictor")
st.markdown("Enter environmental and data quality attributes to predict GHG emissions.")

# Sidebar styling
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3062/3062634.png", width=100)
    st.subheader("Customize Theme")
    st.markdown("Try dark/light mode toggle from Streamlit settings.")
    st.markdown("---")


    st.markdown("**Model Info**")
    st.text("Version: 1.0.0")
    st.text("Trained on: 2025-08-01")
    st.text("Accuracy: 99.6% (R² Score)")
    st.caption("Made using Streamlit")

# --- Dropdown Options ---
substace_options = ["carbon dioxide", "methane", "nitrous oxide", "other GHGs"]
unit_options = ["kg/2018 USD, purchaser price", "kg CO2e/2018 USD, purchaser price"]
source_options = ["Industry", "Commodity"]

# --- Single Prediction UI ---
st.header("🔍 Single Prediction")

selected_substace = st.selectbox("Substace", substace_options)
selected_unit = st.selectbox("Unit", unit_options)
selected_source = st.selectbox("Source", source_options)

supply_chain = st.number_input("Supply Chain Emission Factors without Margins")
margins = st.number_input("Margins of Supply Chain Emission Factors")
reliability = st.number_input("DQ ReliabilityScore of Factors without Margins", min_value=0, max_value=5)
temporal = st.number_input("DQ TemporalCorrelation of Factors without Margins", min_value=0, max_value=5)
geographical = st.number_input("DQ GeographicalCorrelation of Factors without Margins", min_value=0, max_value=5)
technological = st.number_input("DQ TechnologicalCorrelation of Factors without Margins", min_value=0, max_value=5)
datacollection = st.number_input("DQ DataCollection of Factors without Margins", min_value=0, max_value=5)

# --- Make Single Prediction ---
if st.button("Predict Emission"):
    input_data = pd.DataFrame([[
        substace_encoder.transform([selected_substace])[0],
        unit_encoder.transform([selected_unit])[0],
        supply_chain,
        margins,
        reliability,
        temporal,
        geographical,
        technological,
        datacollection,
        source_encoder.transform([selected_source])[0]
    ]], columns=features)

    # Apply scaling
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]
    st.success(f"🌍 Predicted GHG Emission Factor: **{prediction:.4f}**")

# --- Batch Prediction Section ---
st.header("📂 Batch Prediction from File")
uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.write("📝 Preview of uploaded data:")
        st.dataframe(df.head())

        # Encode categorical fields
        df["Substace"] = substace_encoder.transform(df["Substace"])
        df["Unit"] = unit_encoder.transform(df["Unit"])
        df["Source"] = source_encoder.transform(df["Source"])

        # Check if all required features exist
        if not all(col in df.columns for col in features):
            st.error("Missing one or more required columns for prediction.")
        else:
            input_batch = df[features]
            scaled_batch = scaler.transform(input_batch)
            predictions = model.predict(scaled_batch)
            df["Predicted GHG Emission"] = predictions
            st.success("✅ Batch predictions complete.")
            st.dataframe(df)

            csv = df.to_csv(index=False)
            st.download_button("📥 Download Results", csv, file_name="ghg_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
