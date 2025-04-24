import streamlit as st
import os
from src.document_utils import pdf_extract, pdf_chunk, txt_extract, txt_chunk
from src.rag_utils import create_vector_store

def main():
    st.title("Create Vector Store")
    st.write("Upload a PDF or TXT file to create a vector store")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt'])
    
    # Vector store name input
    store_name = st.text_input("Enter vector store name", "my_vector_store")
    
    if uploaded_file and store_name:
        # Create a temporary file to process
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Create Vector Store"):
            try:
                # Determine file type and use appropriate extraction method
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'pdf':
                    st.info("Extracting text from PDF...")
                    text = pdf_extract(temp_path)
                    st.info("Chunking PDF text...")
                    chunks = pdf_chunk(text)
                else:  # txt file
                    st.info("Extracting text from TXT...")
                    text = txt_extract(temp_path)
                    st.info("Chunking TXT text...")
                    chunks = txt_chunk(text)
                
                db_path = f"vector_store/{store_name}"
                
                st.info("Creating vector store...")
                db = create_vector_store(chunks, db_path)
                st.success(f"Vector store created successfully at {db_path}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

if __name__ == "__main__":
    main()
