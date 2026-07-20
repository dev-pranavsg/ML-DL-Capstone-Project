import io
import requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HHAR Classifier", page_icon = "🏃", layout = "wide")
st.title("HHAR Human Activity Classifier")

backend = st.sidebar.text_input("Backend URL", "https://ml-dl-capstone-project.onrender.com").rstrip("/")

try:
    requests.get(f"{backend}/health", timeout = 3).raise_for_status()
    st.sidebar.success("🟢 Backend Connected")
except:
    st.sidebar.error("🔴 Backend Offline")

activities = ["walking", "standing", "sitting", "biking", "stairsup", "stairsdown"]

try:
    activities = requests.get(f"{backend}/classes").json()["activities"]
except:
    pass

if "window" not in st.session_state:
    st.session_state.window = None

gen, upload, paste = st.tabs(["🎲 Generate", "📁 Upload CSV", "⌨️ Paste CSV"])

with gen:
    act = st.selectbox("Activity", activities)
    if st.button("Generate"):
        r = requests.post(f"{backend}/generate", json = {"activity": act})
        if r.ok:
            st.session_state.window = np.array(r.json()["data"])

with upload:
    file = st.file_uploader("CSV", type = "csv")
    if file:
        data = np.loadtxt(file, delimiter = ",")
        if data.shape == (128, 12):
            st.session_state.window = data
        else:
            st.error("CSV must be 128×12")

with paste:
    txt = st.text_area("Paste CSV")
    if st.button("Load"):
        try:
            data = np.loadtxt(io.StringIO(txt), delimiter = ",")
            if data.shape == (128, 12):
                st.session_state.window = data
            else:
                st.error("Input must be 128×12")
        except:
            st.error("Invalid CSV")

if st.session_state.window is not None:
    st.divider()
    st.subheader("Signal Preview")

    df = pd.DataFrame(st.session_state.window, columns = [f"S{i+1}" for i in range(12)])

    st.line_chart(df.iloc[:,:3])

    if st.button("🚀 Predict", type = "primary"):
        r = requests.post(f"{backend}/predict", json = {"data": st.session_state.window.tolist()})

        if r.ok:
            res = r.json()

            c1, c2 = st.columns(2)
            c1.metric("Activity", res["activity"].upper())
            c2.metric("Confidence", f"{res['confidence']:.2%}")

            st.subheader("Probabilities")

            probs = (
                pd.DataFrame(
                    res["all_probabilities"].items(),
                    columns = ["Activity","Probability"]
                )
                .sort_values("Probability", ascending = False)
                .set_index("Activity")
            )

            st.bar_chart(probs)

        else:
            st.error(r.text)
else:
    st.info("Generate a demo signal or upload/paste a CSV.")
