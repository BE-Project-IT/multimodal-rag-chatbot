import torch
torch.classes.__path__ = []

import os
import base64
import streamlit as st
from src.document_processor import DocumentProcessor
from src.embedding_utils import VectorStoreManager
from src.qa_utils import QuestionAnswerer
from src import utils
from src.video_processor import process_uploaded_video
from src.exception_handler import handle_exception

def process_uploaded_file(uploaded_file):
    """Process an uploaded document file."""
    print(f"Processing uploaded file: {uploaded_file.name}")
    filename_base, ext = os.path.splitext(uploaded_file.name)
    ext = ext.lower()[1:]
    vector_store_name = f"{filename_base}_index_faiss"
    
    try:
        # Check if vector store already exists
        if not utils.check_vector_store(uploaded_file.name):
            print(f"No existing vector store found for {uploaded_file.name}")
            
            # Write uploaded file to disk
            print(f"Writing uploaded file to disk as uploaded_document.{ext}")
            with open("uploaded_document." + ext, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Initialize status indicator
            status = st.empty()
            status.info(f"Processing {ext.upper()} document: Initializing...")
            
            # Process the document
            print(f"Creating DocumentProcessor instance...")
            processor = DocumentProcessor()
            
            print(f"Processing document: uploaded_document.{ext}")
            processed_data = processor.process("uploaded_document." + ext, ext)
            
            # Create vector store
            status.info(f"Creating searchable database from processed content...")
            print(f"Creating vector store: {vector_store_name}")
            vectorstore, embeddings = VectorStoreManager.create_vector_store(processed_data, vector_store_name)
            
            status.success(f"{ext.upper()} document processed successfully!")
            print(f"Document processing complete for {uploaded_file.name}")
            
            status.empty()
            return vectorstore, embeddings, vector_store_name
        else:
            # Vector store exists, load it
            print(f"Found existing vector store for {uploaded_file.name}")
            status = st.empty()
            status.info(f"Loading existing database for {uploaded_file.name}...")
            
            print("Getting embeddings model...")
            embeddings = VectorStoreManager.get_embeddings()
            
            print(f"Loading vector store: {vector_store_name}")
            vectorstore = VectorStoreManager.load_vector_store(embeddings, vector_store_name)
            
            status.success(f"Existing database loaded successfully!")
            print(f"Vector store loaded: {vector_store_name}")
            
            status.empty()
            return vectorstore, embeddings, vector_store_name
            
    except Exception as e:
        error_msg = f"Failed to process file {uploaded_file.name}: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(error_msg)
        handle_exception(e)
        return None, None, None

def display_chat_history(file_name):
    """Display the chat history for a specific document."""
    print(f"Displaying chat history for {file_name}")
    chat_history = utils.load_chat_history(file_name)
    print(f"Loaded {len(chat_history)} messages")
    
    chat_container = st.container()
    with chat_container:
        for i, entry in enumerate(chat_history):
            role = entry["role"]
            content = entry["content"]
            print(f"Displaying message {i+1}: role={role}, content={content[:50]}...")
            
            with st.chat_message(role):
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.markdown(content)
                with col2:
                    if role == "user":
                        if st.button("🗑️", key=f"delete_{i}", help="Delete this message"):
                            print(f"Delete button clicked for message {i}")
                            chat_history = utils.delete_message(file_name, i)
                            st.experimental_rerun()
    
    return chat_history

def display_relevant_images(image_list):
    """Display relevant images with navigation controls."""
    if not image_list:
        print("No relevant images to display")
        st.info("No relevant images found.")
        return
        
    print(f"Displaying {len(image_list)} relevant images")
    
    # Initialize image index if not present
    if 'img_index' not in st.session_state:
        print("Initializing image index")
        st.session_state.img_index = 0
        
    # Create layout for image display
    col1, col2, col3 = st.columns([1, 6, 1])
    
    # Previous button
    with col1:
        if st.button("◀️", key="prev_img") and st.session_state.img_index > 0:
            print(f"Previous image button clicked, changing index from {st.session_state.img_index} to {st.session_state.img_index - 1}")
            st.session_state.img_index -= 1
            
    # Display current image
    with col2:
        current_img = image_list[st.session_state.img_index]
        print(f"Displaying image {st.session_state.img_index + 1} of {len(image_list)}")
        st.image(
            base64.b64decode(current_img), 
            caption=f"Image {st.session_state.img_index + 1} of {len(image_list)}"
        )
        
    # Next button
    with col3:
        if st.button("▶️", key="next_img") and st.session_state.img_index < len(image_list) - 1:
            print(f"Next image button clicked, changing index from {st.session_state.img_index} to {st.session_state.img_index + 1}")
            st.session_state.img_index += 1

def main():
    """Main application function."""
    print("Starting Multimodal RAG Chatbot application")
    
    # Configure page
    st.set_page_config(page_title="Multimodal RAG Chatbot", layout="wide")
    
    # Set up sidebar
    st.sidebar.title("Multimodal RAG Chatbot")
    st.sidebar.markdown("Upload a document:")
    uploaded_file = st.sidebar.file_uploader("Supported Documents", type=["pdf", "docx", "pptx", "txt", "xlsx", "jpg", "jpeg", "png", "heic"])
    
    st.sidebar.markdown("Upload a video:")
    uploaded_video = st.sidebar.file_uploader("Supported Video", type=["mp4", "mov", "mkv"])
    
    # Process video if uploaded and button clicked
    if uploaded_video and st.sidebar.button("Process Video"):
        print(f"Video processing button clicked for {uploaded_video.name}")
        
        # Clean up previous data
        utils.cleanup_previous_data()
        
        # Write video to disk
        print(f"Writing video to disk as uploaded_video.mp4")
        with open("uploaded_video.mp4", "wb") as f:
            f.write(uploaded_video.read())
            
        # Process video
        vectorstore, embeddings, vector_store_name = process_uploaded_video("uploaded_video.mp4")
        
        # Update session state
        if vectorstore:
            print("Video processed successfully, updating session state")
            st.session_state.vectorstore = vectorstore
            st.session_state.vector_store_loaded = True
            st.session_state.current_file = uploaded_video.name
            st.session_state.qa_system = utils.initialize_qa_system()
        else:
            print("Video processing failed")
    
    # Initialize session state variables if they don't exist
    for key, default in [
        ('vector_store_loaded', False),
        ('vectorstore', None),
        ('current_file', None),
        ('qa_system', None)
    ]:
        if key not in st.session_state:
            print(f"Initializing session state: {key}={default}")
            st.session_state[key] = default

    # Process document if uploaded
    if uploaded_file is not None:
        print(f"Document file detected: {uploaded_file.name}")
        
        # Check if we need to process a new file
        if st.session_state.current_file != uploaded_file.name:
            print(f"New file detected (current: {st.session_state.current_file}, new: {uploaded_file.name})")
            utils.cleanup_previous_data()
            st.session_state.vector_store_loaded = False
            st.session_state.current_file = uploaded_file.name
            
        # Process file if not already loaded
        if not st.session_state.vector_store_loaded:
            print("Processing document file...")
            vectorstore, embeddings, vector_store_name = process_uploaded_file(uploaded_file)
            
            if vectorstore:
                print("Document processed successfully, updating session state")
                st.session_state.vectorstore = vectorstore
                st.session_state.vector_store_loaded = True
                
        # Initialize QA system if needed
        if st.session_state.qa_system is None:
            print("Initializing QA system...")
            st.session_state.qa_system = utils.initialize_qa_system()
            
            if st.session_state.qa_system is None:
                error_msg = "Failed to initialize the QA system. Please try again."
                print(f"ERROR: {error_msg}")
                st.error(error_msg)
                return

    # If everything is ready, display chat interface
    if st.session_state.vector_store_loaded and st.session_state.qa_system is not None:
        st.subheader("Chat with the Document or Video")
        print("Displaying chat interface")
        
        # Display existing chat history
        chat_history = display_chat_history(st.session_state.current_file)
        
        # Process new questions
        question = st.chat_input("Ask a question:")
        if question:
            print(f"New question received: {question}")
            
            # Reset image index for new question
            st.session_state.img_index = 0
            
            # Clear any previous containers
            prev_answer_container = st.empty()
            prev_answer_container.empty()
            prev_image_container = st.empty()
            prev_image_container.empty()
            
            # Display user question
            with st.chat_message("user"):
                st.markdown(question)
                
            # Add question to chat history
            chat_history.append({"role": "user", "content": question})
            utils.save_chat_history(st.session_state.current_file, chat_history)
            
            # Process question and generate answer
            assistant_container = st.empty()
            with assistant_container.container():
                with st.spinner('Generating answer...'):
                    try:
                        print(f"Processing question through QA system: {question}")
                        answer, relevant_images = st.session_state.qa_system.answer_question(
                            st.session_state.vectorstore, question
                        )
                        print(f"Answer generated, length: {len(answer)}")
                        print(f"Found {len(relevant_images)} relevant images")
                    except Exception as e:
                        error_msg = f"Error generating answer: {str(e)}"
                        print(f"ERROR: {error_msg}")
                        answer = error_msg
                        relevant_images = []
                        handle_exception(e)
                        
            # Display answer
            assistant_container.markdown(answer)
            
            # Add answer to chat history
            chat_history.append({"role": "assistant", "content": answer})
            utils.save_chat_history(st.session_state.current_file, chat_history)
            
            # Display relevant images
            if relevant_images:
                st.subheader("Relevant Image(s)")
                display_relevant_images(relevant_images)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Multimodal RAG Chatbot")
    print("="*50 + "\n")
    main()