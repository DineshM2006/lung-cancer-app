import streamlit as st
import pandas as pd
import joblib
import os

st.title("Lung Cancer Prediction App")
st.write("1 = NO/low, 2 = YES/high. Prob >0.90 = HIGH RISK.")

@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model not found. Run training first.")
        st.stop()

model = load_model()

feature_columns = [
    'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 
    'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 
    'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 
    'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'GENDER_M'
]

def user_input_features():
    GENDER_raw = st.sidebar.selectbox("GENDER (1=F, 2=M)", [1, 2])
    GENDER_M = 0 if GENDER_raw ==1 else 1
    AGE = st.sidebar.slider("AGE", 20, 90, 50)
    SMOKING = st.sidebar.selectbox("SMOKING (1=no, 2=yes)", [1, 2])
    YELLOW_FINGERS = st.sidebar.selectbox("YELLOW_FINGERS (1=no, 2=yes)", [1, 2])
    ANXIETY = st.sidebar.selectbox("ANXIETY (1=no, 2=yes)", [1, 2])
    PEER_PRESSURE = st.sidebar.selectbox("PEER_PRESSURE (1=no, 2=yes)", [1, 2])
    CHRONIC_DISEASE = st.sidebar.selectbox("CHRONIC DISEASE (1=no, 2=yes)", [1, 2])
    FATIGUE = st.sidebar.selectbox("FATIGUE (1=no, 2=yes)", [1, 2])
    ALLERGY = st.sidebar.selectbox("ALLERGY (1=no, 2=yes)", [1, 2])
    WHEEZING = st.sidebar.selectbox("WHEEZING (1=no, 2=yes)", [1, 2])
    ALCOHOL_CONSUMING = st.sidebar.selectbox("ALCOHOL CONSUMING (1=no, 2=yes)", [1, 2])
    COUGHING = st.sidebar.selectbox("COUGHING (1=no, 2=yes)", [1, 2])
    SHORTNESS_OF_BREATH = st.sidebar.selectbox("SHORTNESS OF BREATH (1=no, 2=yes)", [1, 2])
    SWALLOWING_DIFFICULTY = st.sidebar.selectbox("SWALLOWING DIFFICULTY (1=no, 2=yes)", [1, 2])
    CHEST_PAIN = st.sidebar.selectbox("CHEST PAIN (1=no, 2=yes)", [1, 2])

    features_dict = {
        'AGE': AGE,
        'SMOKING': SMOKING,
        'YELLOW_FINGERS': YELLOW_FINGERS,
        'ANXIETY': ANXIETY,
        'PEER_PRESSURE': PEER_PRESSURE,
        'CHRONIC DISEASE': CHRONIC_DISEASE,
        'FATIGUE ': FATIGUE,
        'ALLERGY ': ALLERGY,
        'WHEEZING': WHEEZING,
        'ALCOHOL CONSUMING': ALCOHOL_CONSUMING,
        'COUGHING': COUGHING,
        'SHORTNESS OF BREATH': SHORTNESS_OF_BREATH,
        'SWALLOWING DIFFICULTY': SWALLOWING_DIFFICULTY,
        'CHEST PAIN': CHEST_PAIN,
        'GENDER_M': GENDER_M
    }
    input_df = pd.DataFrame([features_dict])[feature_columns]
    return input_df

input_df = user_input_features()

st.subheader("Features")
st.dataframe(input_df, width="stretch")

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    st.subheader("Result ( >0.90 = YES)")
    if prob > 0.90:
        st.error("YES")
    else:
        st.success("NO")
    st.metric("Probability", f"{prob:.2%}")

