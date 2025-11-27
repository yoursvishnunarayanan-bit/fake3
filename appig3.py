import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# ------------------- PAGE SETTINGS -------------------
st.set_page_config(
    page_title="Fake Profile Detector",
    layout="wide",
    page_icon="🕵️"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #0f2027, #203a43, #2c5364);
    color: white;
}
.main {
    background: transparent;
}
h1 {
    text-align: center;
    font-size: 45px;
    padding-top: 20px;
}
h4 {
    text-align: center;
    color: #cfd8dc;
}

div[data-testid="stSidebar"] {
    background: #0b0b12;
    padding: 20px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #00b4db, #0083b0);
    color: white;
    height: 50px;
    border-radius: 12px;
    font-size: 18px;
}

.result-box {
    background: rgba(0,0,0,0.6);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)


# ------------------- TITLE -------------------
st.markdown("<h1>🕵️ Fake Profile Detector</h1>", unsafe_allow_html=True)
st.markdown("""
<h4>
This tool uses a Hybrid AI Model trained on 10,000 Instagram profiles.  
Simply enter the visible profile details, and the system will analyze forensic patterns.
</h4>
""", unsafe_allow_html=True)


# ------------------- SIDEBAR -------------------
st.sidebar.markdown("## 👤 Profile Information")

username = st.sidebar.text_input("Username", "user123")
full_name = st.sidebar.text_input("Full Name", "User One")
bio = st.sidebar.text_area("Bio / Description", "Living my best life.")

profile_pic = st.sidebar.radio("Profile Pic?", ["Yes", "No"])
external_url = st.sidebar.radio("External URL?", ["Yes", "No"])
private = st.sidebar.radio("Private?", ["Yes", "No"])

followers = st.sidebar.number_input("Followers", min_value=0, value=150)
following = st.sidebar.number_input("Following", min_value=0, value=200)
posts = st.sidebar.number_input("Posts", min_value=0, value=10)

analyze_btn = st.sidebar.button("🔍 Analyze Profile")


# ------------------- FEATURE ENGINEERING -------------------
def extract_features():
    nums_length_username = sum(c.isdigit() for c in username) / (len(username)+1)
    fullname_words = len(full_name.split())
    nums_length_fullname = sum(c.isdigit() for c in full_name) / (len(full_name)+1)

    name_is_username = 1 if full_name.lower() == username.lower() else 0
    desc_length = len(bio)

    return [
        1 if profile_pic == "Yes" else 0,
        nums_length_username,
        fullname_words,
        nums_length_fullname,
        name_is_username,
        desc_length,
        1 if external_url == "Yes" else 0,
        1 if private == "Yes" else 0,
        posts,
        followers,
        following
    ]


# ------------------- LOAD / TRAIN MODEL -------------------
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv("dataset.csv")

        df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

        X = df.drop('fake', axis=1)
        y = df['fake']

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        rf = RandomForestClassifier(n_estimators=150, random_state=42)
        xgb = XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss')

        model = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb)],
            voting='soft'
        )

        model.fit(X, y)
        return model

    except:
        return None

model = train_model()


# ------------------- PREDICTION -------------------
prediction = None
prob = None
if analyze_btn and model is not None:
    features = extract_features()
    final_input = np.array(features).reshape(1, -1)

    prediction = model.predict(final_input)[0]
    prob = model.predict_proba(final_input).max()

if prediction is not None:
    with st.container():
        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        if prediction == 1:
            st.error("🚨 **FAKE PROFILE DETECTED**")
        else:
            st.success("✅ **GENUINE PROFILE DETECTED**")

        st.write("### Confidence:", round(prob * 100, 2), "%")

        st.markdown('</div>', unsafe_allow_html=True)

if analyze_btn:
    if model is None:
        st.error("Model not loaded. Make sure dataset.csv is in the same folder.")
    else:
        features = extract_features()
        final_input = np.array(features).reshape(1, -1)

        prediction
