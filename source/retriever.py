from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv

load_dotenv()

llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
FAISS_PATH = "faiss_index"

def get_qa_chain():

    vectordb = FAISS.load_local(FAISS_PATH, EMBEDDINGS)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain
