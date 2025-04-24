from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from dotenv import load_dotenv
from typing import Dict, List
from langchain.schema import Document

# Load environment variables
load_dotenv()

def text_extract(txt_path: str) -> List[Document]:
    """
    Extracts text from a text file using TextLoader.
    """
    print("Text file is being loaded...")
    loader = TextLoader(txt_path)
    text = loader.load()
    return text

def text_chunk(text: List[Document]) -> List[Document]:
    """
    Splits extracted text into smaller chunks using RecursiveCharacterTextSplitter.
    """
    print("Text is being chunked....")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(text)
    return chunks

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
    Creates or loads a vector store for a given text file and extracts relevant chunks based on a query.
    """
    txt_path, query, db_path = inputs['txt_path'], inputs['query'], inputs['db_path']

    # Create new vector store if it does not exist
    if not os.path.exists(db_path):
        print("Creating a new vector store...\n")
        text = text_extract(txt_path)
        chunks = text_chunk(text)
        db = create_vector_store(chunks, db_path)
    else:
        print("Loading the existing vector store\n")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    relevant_chunks = retrieve_context(db, query)
    context = build_context(relevant_chunks)

    return {'context': context, 'query': query}

def setup_rag_chain():
    template = """ You are an AI model trained for question answering FAQs on a rent payment platform. You should answer the
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

if __name__ == "__main__":
    # Example usage
    TXT_PATH = 'housing_edge_faq.txt'  # Update with your text file path
    current_dir = os.path.join(os.getcwd(), "vector_store")
    persistent_directory = os.path.join(current_dir, "db", "chroma_db_txt")
    
    query = "Someone took my card and used it to pay their rent. What should I do?"
    
    rag_chain = setup_rag_chain()
    answer = rag_chain.invoke({
        'txt_path': TXT_PATH,
        'query': query,
        'db_path': persistent_directory
    })
    
    print(f"Query: {query}\n")
    print(f"Generated answer: {answer}") 