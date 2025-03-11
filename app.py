import streamlit as st


st.title("Triage System")

uploaded_file = st.file_uploader("Upload a video", type = ["mp4", "mov"])
if uploaded_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.video("temp_video.mp4")
    
