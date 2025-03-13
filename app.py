import streamlit as st
import requests
import json

URL = "http://localhost:8000/upload"

st.title("Triage System")

mapper = {
    0: "Normal General Movements",
    1: "Poor repertoire movements",
    2: "Cramped synchronized movements",
    3: "chaotic movements",
    4: "Others"
}

uploaded_file = st.file_uploader("Upload a video", type = ["mp4", "mov"])
if uploaded_file:
    # st.video("temp_video.mp4")
    responseJson = requests.post(URL, files = {"file": uploaded_file.read()})

    response = responseJson.json()
    result = {}
    for index, i in enumerate(response["model_Result"]):
        result[f"{mapper[index]}"] = f"{i:03f}"

    st.write(result)  #TODO retrieve the output value from the corresponding output key of response
