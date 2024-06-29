from langchain_community.document_loaders import PyPDFLoader

file_path = "../arxiv_papers/arxiv-1301.3781.pdf"

loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

print(pages[1])
