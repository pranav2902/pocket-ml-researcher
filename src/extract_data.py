import pinecone
import sys
import time
from langchain_community.document_loaders import PyPDFLoader
# TODO: Check Grobid for document loading
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_pinecone import PineconeVectorStore, Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import hashlib
from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def generate_id(text):
    """
    Generate unique hash based on text, used as id for documents in index
    :param text: Text
    :return: id
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_and_chunk_pdf(code):
    """
    Loads and chunks PDF from local system
    :param code: arxiv paper code
    :return: list of documents making up each pdf
    """
    file = "../arxiv_papers/arxiv-" + code + ".pdf"
    loader = PyPDFLoader(file)
    pages_file = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=400, add_start_index=True
    )
    pdf_splits = text_splitter.split_documents(pages_file)
    return pdf_splits


def create_pinecone_index(index_name: str, dim: int):
    """
    Creates Pinecone index
    :param index_name: Name of index
    :param dim: Dimension of vector embeddings generated (check embedding model for info)
    :return:
    """
    # Create pinecone index
    pc = Pinecone()
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    pc_index = pc.Index(index_name)
    return pc_index


def is_document_stored(pc_index: pinecone.Index, doc) -> bool:
    """
    Check if document already exists in Index
    :param pc_index: Pinecone index variable
    :param doc: document
    :return: bool
    """
    text = doc.page_content
    id = generate_id(text)
    print(id)
    response = pc_index.query(id=id, top_k=1)
    print(response)
    if not len(response["matches"]):
        return False
    elif id in response["matches"][0]["id"]:
        print("Document exists in Index, returning")
        return True

def init_vectorstore(docs_to_be_upserted: list[Document], metadata_list: list[dict], index_name: str,
                     embeddings: HuggingFaceEmbeddings) -> PineconeVectorStore:
    """
    Initialize vectorstore
    :param docs_to_be_upserted: New documents to be upserted
    :param metadata_list: List of metadata for the documents
    :param index_name: Name of existing pinecone index
    :param embeddings: Embedding model (HFEmbeddings object)
    :return: Pinecone Vector Store
    """
    texts = [d.page_content for d in docs_to_be_upserted]
    ids = [generate_id(text) for text in texts]
    vs = PineconeVectorStore.from_texts(texts, index_name=index_name, embedding=embeddings,ids=ids,
                                        metadatas=metadata_list)
    print("Completed upsert of documents")
    return vs


def retrieve_docs(query: str, retriever):
    """
    Test retrieval of relevant documents given query
    :param query: Query used for retrieval
    :param retriever: Retriever
    :return:
    """
    retrieved_docs = retriever.invoke(query)
    print(retrieved_docs)


def format_docs(docs) -> str:
    """
    Function to join documents together into one string
    :param docs: List of documents
    :return: Joined string
    """
    return "\n\n".join(doc.page_content for doc in docs)


def generate_prompt(prompt_template=None) -> ChatPromptTemplate:
    prompt_template = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer 
            the question. If you don't know the answer, say that you 
            don't know. Explain in detail when asked, otherwise stay concise"

            {context}

            Question: {question}

            Answer:""" if not prompt_template else prompt_template

    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt


if __name__ == "__main__":
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    file_path = "../arxiv_papers/arxiv-1301.3781"
    codes = ["1301.3781",
             "1312.6114",
             "1409.0473",
             "1412.6980",
             "1505.04597",
             "1512.03385",
             "1810.04805",
             "2010.11929",
             "2109.01652",
             "2109.11978",
             "2109.13226"]

    dims = 384
    index_name = "pdf-store-1"
    logger.info("Creating Pinecone Index")
    index = create_pinecone_index(index_name, dims)
    metadata_list, new_docs = [], []
    for code in codes:
        logger.info(f"Loading and Chunking PDF code: {code}")
        splits = load_and_chunk_pdf(code)
        for doc in splits:
            if not is_document_stored(index,doc):
                logger.info(f"Document is not stored, to upsert - code: {code}")
                metadata = {"code": code}
                metadata_list.append(metadata)
                new_docs.append(doc)

    print(metadata_list)
    if new_docs:
        vs = init_vectorstore(new_docs, metadata_list, index_name, embeddings_model)
    else:
        # Check working of is_document_stored function
        print("No New documents to be upserted")
        vs = PineconeVectorStore(index_name=index_name,embedding=embeddings_model)
    # Add support for loading w Arxiv code alone
    logger.info("Testing retriever")
    test_query = "Explain the architecture of an NNLM briefly"
    top_k = 3
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    retrieve_docs(test_query, retriever)

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = generate_prompt()

    custom_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                    )

    for resp in custom_chain.stream(test_query):
        print(resp, end="", flush=True)
    sys.exit()
