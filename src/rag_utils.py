from typing import Dict, List
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from .document_utils import pdf_extract, pdf_chunk, txt_extract, txt_chunk

def create_vector_store(chunks: List[Document], db_path: str) -> Chroma:
    """
    Creates a Chroma vector store from chunked documents.
    """
    print("Chrome vector store is created...\n")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=db_path)
    return db

def retrieve_context(db: Chroma, query: str) -> List[Document]:
    """
    Retrieves relevant document chunks from the Chroma vector store based on a query.
    """
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    print("Relevant chunks are retrieved...\n")
    relevant_chunks = retriever.invoke(query)
    return relevant_chunks

def build_context(relevant_chunks: List[Document]) -> str:
    """
    Builds a context string from retrieved relevant document chunks.
    """
    print("Context is built from relevant chunks")
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    return context

def get_context(inputs: Dict[str, str]) -> Dict[str, str]:
    """
    Loads an existing vector store and retrieves relevant chunks based on a query.
    Raises FileNotFoundError if the vector store does not exist.
    """
    query, db_path = inputs['query'], inputs['db_path']

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Vector store not found at path: {db_path}. Please create the vector store first.")

    print("Loading the vector store ...")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    relevant_chunks = retrieve_context(db, query)
    context = build_context(relevant_chunks)

    return {'context': context, 'query': query}

def setup_rag_chain():
    template = """ You are an AI model trained for FAQs on a rent payment platform. You should answer the
    given question based on the given context only.
    Question : {query}
    \n
    Context : {context}
    \n
    If the answer is not present in the given context, respond as: The answer to this question is not available
    in the provided content.
    """

    rag_prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model='gpt-4o-mini')
    str_parser = StrOutputParser()

    rag_chain = (
        RunnableLambda(get_context)
        | rag_prompt
        | llm
        | str_parser
    )
    
    return rag_chain