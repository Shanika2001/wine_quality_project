import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

st.set_page_config(page_title="Wine Quality Predictor", layout="wide")


@st.cache_data
def load_data(path="data/WineQT.csv"):
    df = pd.read_csv(path)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])  
    return df


@st.cache_resource
def load_model_and_scaler(model_path="models/model.pkl", scaler_path="models/scaler.pkl"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# --- Climate Guess ---
def climate_guess(alcohol, acidity):
    label = "Moderate climate style 🌤️"
    if alcohol > 13:
        label = "Likely hot climate style 🌞"
    elif alcohol < 11:
        label = "Likely cool climate style ❄️"
    if acidity >= 8 and alcohol <= 13:
        label = "Likely cool climate style ❄️ (high acidity)"
    return label

st.title("🍷 Wine Quality Predictor")
menu = st.sidebar.selectbox("Navigation", ["Home", "Data Exploration", "Visualizations", "Prediction", "Model Performance"])

data = load_data()
model, scaler = load_model_and_scaler()
expected_features = list(scaler.feature_names_in_)


if menu == "Home":
    st.header("Welcome to the Wine Quality Predictor")
    st.markdown("""
    This app predicts wine quality using two modes:
    - **Simple Prediction**: For normal users (no lab measurements needed).
    - **Full Prediction**: For lab workers with detailed wine chemical properties.
    """)

    st.subheader("🍷 Quick Simple Prediction")
    wine_type = st.selectbox("Wine Type", ["Red", "White"], key="simple_wine_type")
    sweetness = st.select_slider("Sweetness Level", ["Dry", "Off-dry", "Medium", "Sweet"], key="simple_sweetness")
    body = st.radio("Body (Mouthfeel)", ["Light", "Medium", "Full"], key="simple_body")
    acidity_level = st.radio("Acidity Level (Taste)", ["Low", "Medium", "High"], key="simple_acidity")
    alcohol = st.slider("Alcohol % (From Label)", 8.0, 16.0, 12.5, key="simple_alcohol")

    if st.button("Predict quality", key="simple_predict_btn"):
        mapping = {}
        mapping['fixed acidity'] = {'Low': 5.0, 'Medium': 7.0, 'High': 9.0}[acidity_level]
        mapping['volatile acidity'] = {'Red': 0.6, 'White': 0.3}[wine_type]
        mapping['citric acid'] = {'Low': 0.2, 'Medium': 0.4, 'High': 0.6}[acidity_level]
        mapping['residual sugar'] = {'Dry': 2.0, 'Off-dry': 5.0, 'Medium': 10.0, 'Sweet': 20.0}[sweetness]
        mapping['chlorides'] = {'Red': 0.08, 'White': 0.05}[wine_type]
        mapping['free sulfur dioxide'] = 30.0 if wine_type == "White" else 15.0
        mapping['total sulfur dioxide'] = 120.0 if wine_type == "White" else 60.0
        mapping['density'] = 0.995 if body == "Light" else (0.997 if body == "Medium" else 1.0)
        mapping['pH'] = {'Low': 3.6, 'Medium': 3.3, 'High': 3.0}[acidity_level]
        mapping['sulphates'] = {'Red': 0.65, 'White': 0.45}[wine_type]
        mapping['alcohol'] = alcohol
        mapping['type_white'] = 1 if wine_type == "White" else 0

        x_df = pd.DataFrame([mapping])

        
        for col in expected_features:
            if col not in x_df.columns:
                x_df[col] = 0
    
        x_df = x_df[expected_features]

        try:
            x_scaled = scaler.transform(x_df)
            pred_proba = model.predict_proba(x_scaled)[0]
            pred_class = model.predict(x_scaled)[0]
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        st.success(f"Predicted quality class: {label_map.get(pred_class, 'Unknown')}")
        st.write(pd.DataFrame({'class': [label_map[i] for i in range(len(pred_proba))],
                               'probability': pred_proba}))

        climate = climate_guess(alcohol, mapping['fixed acidity'])
        st.info(f"Climate-style guess: {climate}")


elif menu == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(data.shape)
    st.write("Columns:", list(data.columns))
    st.dataframe(data.head())

elif menu == "Visualizations":
    st.subheader("Quality Distribution")
    fig = px.histogram(data, x='quality')
    st.plotly_chart(fig)

    st.subheader("Alcohol vs. Quality")
    fig2 = px.box(data, x='quality', y='alcohol')
    st.plotly_chart(fig2)

    st.subheader("Correlation Heatmap")
    corr = data.corr(numeric_only=True)
    fig3, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig3)


elif menu == "Prediction":
    st.header("Full Prediction (Laboratory Inputs)")
    numeric_cols = [c for c in data.columns if data[c].dtype != 'object' and c != 'quality']
    inputs = {}
    colA, colB = st.columns(2)
    for i, col_name in enumerate(numeric_cols):
        mn = float(data[col_name].min())
        mx = float(data[col_name].max())
        mean = float(data[col_name].mean())
        if i % 2 == 0:
            inputs[col_name] = colA.slider(col_name, mn, mx, mean)
        else:
            inputs[col_name] = colB.number_input(col_name, min_value=mn, max_value=mx, value=mean)

    if st.button("Predict quality (Full Inputs)"):
        x_df = pd.DataFrame([inputs])

        
        for col in expected_features:
            if col not in x_df.columns:
                x_df[col] = 0
        
        x_df = x_df[expected_features]

        try:
            x_scaled = scaler.transform(x_df)
            pred_proba = model.predict_proba(x_scaled)[0]
            pred_class = model.predict(x_scaled)[0]
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        label_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        st.success(f"Predicted quality class: {label_map.get(pred_class, 'Unknown')}")
        st.write(pd.DataFrame({'class': [label_map[i] for i in range(len(pred_proba))],
                               'probability': pred_proba}))

        alcohol = inputs.get('alcohol', None)
        acidity = inputs.get('fixed acidity', None)
        climate = climate_guess(alcohol, acidity)
        st.info(f"Climate-style guess: {climate}")


elif menu == "Model Performance":
    st.subheader("Model Performance Metrics")
    try:
        metrics_df = pd.read_csv("models/model_metrics.csv", index_col=0)
        st.dataframe(metrics_df)
    except:
        st.warning("Model metrics file not found.")
