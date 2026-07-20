import streamlit as st
import numpy as np
import requests

st.title("HHAR Human Activity Classifier")

backend_url = st.sidebar.text_input("Backend API URL", value = "https://ml-dl-capstone-project.onrender.com")

if st.button("Generate & Test Random Window"):
    t = np.linspace(0, 4 * np.pi, 128)
    mock_matrix = np.zeros((128, 12))
    
    for i in range(12):
        if i % 3 == 0:
            mock_matrix[:, i] = np.sin(t * (1 + (i % 3)))
        elif i % 3 == 1:
            mock_matrix[:, i] = np.cos(t * (1.5 + (i % 2)))
        else:
            mock_matrix[:, i] = np.sin(t) * np.cos(t * 2) + np.random.normal(0, 0.1, 128)
            
    test_window = mock_matrix.tolist()
    
    with st.spinner("Sending payload to API..."):
        try:
            clean_url = f"{backend_url.rstrip('/')}/predict"
            res = requests.post(clean_url, json = {"data": test_window})
            
            if res.status_code == 200:
                result = res.json()
                st.metric("Predicted State", result["activity"].upper())
                st.metric("Confidence Score", f"{result['confidence']:.2%}")
                st.bar_chart(result["all_probabilities"])
            else:
                st.error(f"API Error: {res.text}")
        except Exception as e:
            st.error(f"Could not connect to backend server: {e}")
