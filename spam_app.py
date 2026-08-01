import streamlit as st
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
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

    if not email.strip():
        st.warning("Please enter an email.")
    else:
        transformed_email = vectorizer.transform([email])

        prediction = model.predict(transformed_email)[0]

        # Handle numeric or string labels
        if prediction == 1 or str(prediction).lower() == "spam":
            st.error("🚨 Spam Email Detected")
        else:
            st.success("✅ Not Spam")

        # Show confidence if supported
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(transformed_email)[0]
            confidence = max(probability) * 100

            st.progress(confidence / 100)
            st.write(f"**Confidence:** {confidence:.2f}%")