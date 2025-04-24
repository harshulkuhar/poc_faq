from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf_extract(pdf_path: str) -> List[Document]:
    """
    Extracts text from a PDF file using PyPDFLoader.
    """
    print("PDF file text is extracted...")
    loader = PyPDFLoader(pdf_path)
    pdf_text = loader.load()
    return pdf_text

def pdf_chunk(pdf_text: List[Document]) -> List[Document]:
    """
    Splits extracted PDF text into smaller chunks using RecursiveCharacterTextSplitter.
    """
    print("PDF file text is chunked....")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pdf_text)
    return chunks

def txt_extract(txt_path: str) -> List[Document]:
    """
    Extracts text from a txt file using TextLoader.
    """
    print("txt file is being loaded...")
    loader = TextLoader(txt_path)
    text = loader.load()
    return text

def txt_chunk(text: List[Document]) -> List[Document]:
    """
    Splits extracted text into smaller chunks using RecursiveCharacterTextSplitter.
    """
    print("Text is being chunked....")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(text)
    return chunks