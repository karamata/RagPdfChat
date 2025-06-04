import streamlit as st
import os
from typing import List, Dict, Any
import time
import uuid

from rag_system import RAGSystem
from pdf_processor import PDFProcessor
from pinecone_manager import PineconeManager
from mistral_client import MistralClient

# Set page configuration
st.set_page_config(
    page_title="RAG Chat Application",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "pinecone_index_name" not in st.session_state:
    st.session_state.pinecone_index_name = f"rag-index-{str(uuid.uuid4())[:8]}"

def initialize_rag_system():
    """Initialize the RAG system with API keys"""
    try:
        # Get API keys from environment variables with fallbacks
        mistral_api_key = os.getenv("MISTRAL_API_KEY", "KjRVvzvLrjWnIGVwFuWt3i3iig5gvyNY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY", "pcsk_2s3Nn6_5HvwZFers9bidgr5ikkqVhbUC32tjyRi5UzyvvinjKgBdtnzKtnAXo3wVXKcAWe")
        
        if not st.session_state.rag_system:
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_system = RAGSystem(
                    mistral_api_key=mistral_api_key,
                    pinecone_api_key=pinecone_api_key,
                    index_name=st.session_state.pinecone_index_name
                )
        return True
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return False

def process_uploaded_files(uploaded_files):
    """Process uploaded PDF files"""
    if not uploaded_files:
        return
    
    if not st.session_state.rag_system:
        if not initialize_rag_system():
            return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (idx) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Check if file is already processed
            if uploaded_file.name in st.session_state.processed_files:
                status_text.text(f"Skipping {uploaded_file.name} (already processed)")
                continue
            
            # Read file content
            file_content = uploaded_file.read()
            
            # Process the PDF
            success = st.session_state.rag_system.add_document(
                file_content=file_content,
                filename=uploaded_file.name
            )
            
            if success:
                st.session_state.processed_files.append(uploaded_file.name)
                st.success(f"Successfully processed {uploaded_file.name}")
            else:
                st.error(f"Failed to process {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

def main():
    """Main application function"""
    st.title("📚 RAG Chat Application")
    st.markdown("Upload PDF documents and chat with their content using AI")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to add to the knowledge base"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s)")
            
            if st.button("Process Files", type="primary"):
                process_uploaded_files(uploaded_files)
        
        # Show processed files
        if st.session_state.processed_files:
            st.subheader("Processed Documents")
            for filename in st.session_state.processed_files:
                st.text(f"✅ {filename}")
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    if not st.session_state.processed_files:
        st.info("Please upload and process PDF documents to start chatting!")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if not st.session_state.rag_system:
                        if not initialize_rag_system():
                            st.error("RAG system not initialized. Please check your configuration.")
                            return
                    
                    response = st.session_state.rag_system.query(prompt)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
