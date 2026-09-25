import streamlit as st
import pickle
from PIL import Image

st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🫁", layout="centered")

# ---------- CUSTOM CSS (background, fonts, cards, buttons) ----------
st.markdown("""
<style>
/* Page background - gradient */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #f5f5f5;
}

/* Title styling */
h1 {
    text-align: center;
    color: #ffffff;
    font-weight: 700;
    text-shadow: 0px 2px 6px rgba(0,0,0,0.4);
}

/* Card-style container for inputs */
.block-container {
    padding-top: 2rem;
    max-width: 700px;
}

/* Number input styling */
div[data-baseweb="input"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 10px;
}

/* Predict button */
div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.6em 2em;
    transition: 0.3s;
}
div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0px 4px 15px rgba(0,114,255,0.5);
}

/* Result box */
.result-box {
    padding: 1em;
    border-radius: 12px;
    text-align: center;
    font-size: 1.2em;
    font-weight: 600;
    margin-top: 1em;
}
.safe { background-color: rgba(46, 204, 113, 0.2); border: 1px solid #2ecc71; color: #2ecc71; }
.danger { background-color: rgba(231, 76, 60, 0.2); border: 1px solid #e74c3c; color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


# ---------- APP LOGIC ----------
def main():
    st.title("🫁 Lung Cancer Prediction")

    img = Image.open('Lung.png')
    st.image(img, width=700)

    st.markdown("### Enter Patient Details")
    age = st.number_input('Enter age')
    smoke = st.number_input('Enter smoke level')
    area = st.number_input('Enter area quality')
    alcohol = st.number_input('Alcohol consumption')

    feature = [age, smoke, area, alcohol]

    knn = pickle.load(open('knn_model.sav', 'rb'))
    scaler = pickle.load(open('scaler_knn.sav', 'rb'))

    pred = st.button('Predict')
    if pred:
        result = knn.predict(scaler.transform([feature]))
        result = int(result[0])
        if result == 0:
            st.markdown('<div class="result-box safe">✅ Not a Cancer Patient</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box danger">⚠️ Cancer Patient</div>', unsafe_allow_html=True)


main()

