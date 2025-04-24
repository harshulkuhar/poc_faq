import os
import streamlit as st
from src.rag_utils import setup_rag_chain

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = setup_rag_chain()

def main():
    st.title("Pay On Credit FAQ Chatbot")
    
    # Initialize session state
    initialize_session_state()
    
    # Vector store path setup
    current_dir = os.path.join(os.getcwd(), "vector_store")
    persistent_directory = os.path.join(current_dir, "housing_edge_faq")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({
                    'query': prompt,
                    'db_path': persistent_directory
                })
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Add a clear chat button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main() 