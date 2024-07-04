import streamlit as st
import requests

st.title("Pocket ML Researcher")
st.subheader("Ask a question, or add a Research Paper to your Knowledge Base")

st.header("Add an arxiv paper")
code = st.text_input("Enter Arxiv code")
if st.button("Add paper"):
    response = requests.post("http://localhost:8000/add-paper/",json={"code":code})
    if response.status_code == 200:
        st.success("Paper added to knowledge base successfully!")
    else:
        st.error("Failed to add paper")

st.header("Ask a question")
query = st.text_input("Enter question")
if st.button("Ask"):
    with st.spinner("Processing Response ..."):
        response = requests.post("http://localhost:8000/ask-question/", json={"question": query}, stream=True)
        if response.status_code == 200:
            content=""
            # resp_line is a placeholder that fixes an element on streamlit.
            # resp_line.write will append tokens to the same element space on front-end, instead of
            # repeating the tokens each time on the front-end
            resp_line = st.empty()
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("output:"):
                        content+=decoded_line[7:]
                        resp_line.write(content)
        else:
            st.error("Failed to get Response")

