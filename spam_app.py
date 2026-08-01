import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detector")
st.write("Enter an email message below to check whether it is Spam or Not Spam.")

email = st.text_area("Enter Email")

if st.button("Detect Spam"):

    if email.strip() == "":
        st.warning("Please enter an email.")
    else:

        transformed_email = vectorizer.transform([email])

        prediction = model.predict(transformed_email)

        if prediction[0] == "spam" or prediction[0] == 1:
            st.error("🚨 Spam Email Detected")
        else:
            st.success("✅ Not Spam")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(transformed_email)

            confidence = probability.max() * 100

            st.progress(float(confidence / 100))

            st.write(f"Confidence : {confidence:.2f}%")