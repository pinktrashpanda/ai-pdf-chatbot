import streamlit as st
import requests

st.title("AI PDF Chatbot")

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if not st.session_state.uploaded:
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            files = {"file": uploaded_file.getvalue()}
            res = requests.post("http://localhost:8000/upload_pdf/", files=files)
        if res.status_code == 200:
            st.session_state.uploaded = True
            st.success("PDF processed! Ask your questions below.")
        else:
            st.error("Failed to process PDF.")
else:
    query = st.text_input("Ask a question about your PDF:")
    if st.button("Ask") and query:
        with st.spinner("Thinking..."):
            res = requests.post("http://localhost:8000/ask/", data={"query": query})
        if res.status_code == 200:
            answer = res.json()["answer"]
            st.write("**Answer:**", answer)
            with st.expander("Show context"):
                st.write(res.json()["context"])
        else:
            st.error("Failed to get answer.")