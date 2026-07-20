import streamlit as st
import numpy as np
import requests

st.title("HHAR Human Activity Classifier")

backend_url = st.sidebar.text_input("Backend API URL",  value = "https://ml-dl-capstone-project.onrender.com")

if st.button("Generate & Test Random Window"):
    test_window = np.random.randn(128, 12).tolist()
    
    with st.spinner("Sending payload to API..."):
        try:
            res = requests.post(f"{backend_url}/predict", json = {"data": test_window})
            if res.status_code == 200:
                result = res.json()
                st.metric("Predicted State", result["activity"].upper())
                st.metric("Confidence Score", f"{result['confidence']:.2%}")
                st.bar_chart(result["all_probabilities"])
            else:
                st.error(f"API Error: {res.text}")
        except Exception as e:
            st.error(f"Could not connect to backend server: {e}")
