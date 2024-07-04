from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from extract_data import *
from fastapi.responses import StreamingResponse

app = FastAPI()


class ArxivPaper(BaseModel):
    code: str


class Query(BaseModel):
    question: str


@app.on_event("startup")
async def startup_event():
    global embeddings_model
    global vectorstore
    global llm
    global index_name
    index_name = "pdf-store-1"
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings_model)
    llm = ChatOpenAI(model="gpt-3.5-turbo")


@app.post("/add-paper")
async def add_arxiv_paper_to_knowledge_base(paper: ArxivPaper):
    global vectorstore
    vectorstore = load_pdf_into_index([paper.code], index_name)
    if vectorstore:
        return {"message": "Paper added successfully"}
    else:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings_model)
        raise HTTPException(status_code=500, detail="Paper already exists in Vectorstore")


@app.post("/ask-question/")
async def ask_query(question: Query):
    async def response_generator():
        try:
            async for chunk in return_response(vectorstore, question.question, llm=llm, top_k=3):
                logging.info(f"Streaming chunk: {chunk}")
                yield f"output:{chunk}\n\n"
        except Exception as e:
            logging.error(f"Error:{str(e)}")
            yield f"output:Error: {str(e)}\n\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
