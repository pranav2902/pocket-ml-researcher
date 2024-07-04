# pocket-ml-researcher

This project aims to develop a RAG powered assistant to answer questions on various scientific research papers, currently limited to AI research.
A ChatGPT-3.5 LLM is utilized, and augmented using Research papers written after 2022, to improve it's knowledge. This application is served through FastAPI and Streamlit.

### Add Research Papers to Knowledge Base
Currently, 10 research papers from Arxiv are in the Knowledge base. You have the option to add more papers.

Currently, only adding Arxiv papers is supported.

### Query your Knowledge Base
You can ask the chatbot questions to understand any AI concepts. 

### Directions
To launch app, run the following commands from root directory:
```
pip install -r requirements.txt
cd src
uvicorn app:app --reload
streamlit run streamlit_app.py
```
