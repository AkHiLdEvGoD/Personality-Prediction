import streamlit as st
import requests

st.title("🧠 Personality Prediction App")

st.markdown("Fill out the form to predict if you're an extrovert or introvert")

# Take input
time_spend_alone = st.slider("Time spent alone (0–11)", 0, 11, 5)
stage_fear = st.selectbox("Do you have stage fear?", ["Yes", "No"])
social_event_attendance = st.slider("Social event attendance (0–10)", 0, 10, 5)
going_outside = st.slider("Frequency of going outside (0–7)", 0, 7, 3)
drained_after_socializing = st.selectbox("Do you feel drained after socializing?", ["Yes", "No"])
friends_circle_size = st.slider("Number of close friends (0–15)", 0, 15, 5)
post_frequency = st.slider("Social media post frequency (0–10)", 0, 10, 4)

if st.button("Predict"):
    # Create payload
    payload = {
        "time_spend_alone": time_spend_alone,
        "stage_fear": stage_fear ,
        "social_event_attendance": social_event_attendance,
        "going_outside": going_outside,
        "drained_after_socializing": drained_after_socializing ,
        "friends_circle_size": friends_circle_size,
        "post_frequency": post_frequency
    }

    # Send request to FastAPI backend
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"🎯 Predicted Personality: **{result['Predicted Personality']}**")
        else:
            st.error(f"❌ Error: {response.json()['detail']}")
    except Exception as e:
        st.error(f"🚨 Could not reach the backend. Make sure FastAPI is running.\n\nError: {e}")
