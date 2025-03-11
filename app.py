import streamlit as st
import requests

URL = "localhost:8000/upload"

st.title("Triage System")

uploaded_file = st.file_uploader("Upload a video", type = ["mp4", "mov"])
if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video("temp_video.mp4")
    with open("temp_video.mp4", "rb") as f:
        response = requests.post(URL, {"file": {f}})

    st.write(response)
    