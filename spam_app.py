import streamlit as st
import pickle
import os

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detector")
st.write("Enter an email message below to check whether it is Spam or Not Spam.")

# Check if required files exist
if not os.path.exists("model.pkl"):
    st.error("❌ model.pkl not found.")
    st.stop()

if not os.path.exists("vectorizer.pkl"):
    st.error("❌ vectorizer.pkl not found.")
    st.stop()

# Load model and vectorizer
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

except Exception as e:
    st.error(f"❌ Error loading model files:\n\n{e}")
    st.stop()

# User input
email = st.text_area("Enter Email")

if st.button("Detect Spam"):

    if email.strip() == "":
        st.warning("Please enter an email.")
    else:
        try:
            # Transform text
            transformed_email = vectorizer.transform([email])

            # Predict
            prediction = model.predict(transformed_email)[0]

            # Display result
            if prediction == 1 or str(prediction).lower() == "spam":
                st.error("🚨 Spam Email Detected")
            else:
                st.success("✅ Not Spam")

            # Confidence score
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(transformed_email)[0]
                confidence = max(probs) * 100

                st.subheader("Prediction Confidence")
                st.progress(float(confidence) / 100)
                st.write(f"**Confidence:** {confidence:.2f}%")

        except Exception as e:
            st.error(f"❌ Prediction Error:\n\n{e}")