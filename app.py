import streamlit as st
import numpy as np
import requests

st.title("HHAR Human Activity Classifier")

backend_url = st.sidebar.text_input("Backend API URL", value = "https://ml-dl-capstone-project.onrender.com")

if st.button("Generate & Test Random Window"):
    t = np.linspace(0, 4 * np.pi, 128)
    mock_matrix = np.zeros((128, 12))
    
    profile = np.random.choice(['low_freq', 'high_freq', 'mixed_phase', 'heavy_variance'])
    
    for i in range(12):
        if profile == 'low_freq':
            freq = 0.5 + (i * 0.1)
            phase = np.random.uniform(0, np.pi)
            mock_matrix[:, i] = np.sin(t * freq + phase)
            
        elif profile == 'high_freq':
            freq = 3.0 + (i * 0.2)
            mock_matrix[:, i] = np.cos(t * freq)
            
        elif profile == 'mixed_phase':
            freq_a = 1.2 if i % 2 == 0 else 2.5
            mock_matrix[:, i] = np.sin(t * freq_a) * np.cos(t * 0.8)
            
        else: 
            mock_matrix[:, i] = np.sin(t * 1.5) + np.random.normal(0, 0.4, 128)
            
    test_window = mock_matrix.tolist()
    
    with st.spinner("Sending payload to API..."):
        try:
            clean_url = f"{backend_url.rstrip('/')}/predict"
            res = requests.post(clean_url, json = {"data": test_window})
            
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
