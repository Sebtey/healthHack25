import streamlit as st
import requests

URL = "http://localhost:8000/upload"

st.title("Triage System")

uploaded_file = st.file_uploader("Upload a video", type = ["mp4", "mov"])
if uploaded_file:
    st.video("temp_video.mp4")
    responseJson = requests.post(URL, files = {"file": uploaded_file.read()})

    response = responseJson.json()

    st.write(response) #TODO retrieve the output value from the corresponding output key of response
    