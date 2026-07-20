import streamlit as st
import numpy as np
import requests
import os

st.title("HHAR Human Activity Classifier")
backend_url = st.sidebar.text_input("Backend API URL", value = "https://ml-dl-capstone-project.onrender.com")

@st.cache_data
def load_real_data():
    return np.load("demo_windows.npy") if os.path.exists("demo_windows.npy") else None

real_windows = load_real_data()

if st.button("Generate & Test Random Window"):
    if real_windows is not None:
        test_window = real_windows[np.random.randint(len(real_windows))].tolist()
    else:
        st.warning("'demo_windows.npy' not found in repo root. Using fallback wave data.")
        t = np.linspace(0, 4 * np.pi, 128)
        mock_matrix = np.zeros((128, 12))
        for i in range(12):
            mock_matrix[:, i] = np.sin(t * (1 + (i % 2))) + np.random.normal(0, 0.1, 128)
        test_window = mock_matrix.tolist()
    
    with st.spinner("Sending payload to API..."):
        try:
            res = requests.post(f"{backend_url.rstrip('/')}/predict", json = {"data": test_window})
            if res.status_code == 200:
                result = res.json()
                st.metric("Predicted State", result["activity"].upper())
                st.metric("Confidence Score", f"{result['confidence']:.2%}")
                st.subheader("Model Class Probabilities")
                st.bar_chart(result["all_probabilities"])
            else:
                st.error(f"API Error: {res.text}")
        except Exception as e:
            st.error(f"Could not connect to backend server: {e}")
