import os
import json
import shutil
import streamlit as st
from src.qa_utils import QuestionAnswerer
from src.exception_handler import handle_exception

def cleanup_previous_data():
    """Clean up any previous temporary data for fresh processing."""
    try:
        print("Cleaning up previous data...")
        
        # Clean up figures directory
        if os.path.exists("figures"):
            print("Removing contents of figures directory...")
            for file in os.listdir("figures"):
                file_path = os.path.join("figures", file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            os.makedirs("figures", exist_ok=True)
            print("Created figures directory")
        
        # Remove temporary uploaded files
        for file in os.listdir("."):
            if file.startswith("uploaded_"):
                try:
                    os.unlink(file)
                    print(f"Deleted temporary file: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
                    
        print("Cleanup complete")
        
    except Exception as e:
        error_msg = f"Error during cleanup: {str(e)}"
        print(f"ERROR: {error_msg}")
        handle_exception(e)

def check_vector_store(filename):
    """Check if a vector store exists for the given filename."""
    try:
        base_name = os.path.splitext(filename)[0]
        store_path = f"{base_name}_index_faiss"
        
        print(f"Checking for existing vector store at: {store_path}")
        exists = os.path.exists(store_path) and os.path.isdir(store_path)
        
        if exists:
            print(f"Vector store found: {store_path}")
        else:
            print(f"Vector store not found: {store_path}")
            
        return exists
        
    except Exception as e:
        error_msg = f"Error checking vector store: {str(e)}"
        print(f"ERROR: {error_msg}")
        handle_exception(e)
        return False

def initialize_qa_system():
    """Initialize the question answering system."""
    try:
        print("Initializing question answering system...")
        status = st.empty()
        status.info("Initializing question answering system...")
        
        qa_system = QuestionAnswerer()
        
        print("Question answering system initialized successfully")
        status.success("Question answering system ready!")
        status.empty()
        
        return qa_system
        
    except Exception as e:
        error_msg = f"Error initializing QA system: {str(e)}"
        print(f"ERROR: {error_msg}")
        st.error(error_msg)
        handle_exception(e)
        return None

def get_chat_history_path(file_name):
    """Get the path to the chat history file for a specific document."""
    try:
        base_name = os.path.splitext(file_name)[0]
        history_dir = "chat_history"
        os.makedirs(history_dir, exist_ok=True)
        
        history_path = os.path.join(history_dir, f"{base_name}_history.json")
        print(f"Chat history path: {history_path}")
        
        return history_path
        
    except Exception as e:
        error_msg = f"Error getting chat history path: {str(e)}"
        print(f"ERROR: {error_msg}")
        handle_exception(e)
        return None

def load_chat_history(file_name):
    """Load chat history for a specific document."""
    try:
        history_path = get_chat_history_path(file_name)
        
        if history_path and os.path.exists(history_path):
            print(f"Loading existing chat history from: {history_path}")
            with open(history_path, "r") as f:
                history = json.load(f)
                print(f"Loaded {len(history)} chat messages")
                return history
        else:
            print("No existing chat history found, creating new history")
            return []
            
    except Exception as e:
        error_msg = f"Error loading chat history: {str(e)}"
        print(f"ERROR: {error_msg}")
        handle_exception(e)
        return []

def save_chat_history(file_name, chat_history):
    """Save chat history for a specific document."""
    try:
        history_path = get_chat_history_path(file_name)
        
        if history_path:
            print(f"Saving chat history with {len(chat_history)} messages to: {history_path}")
            with open(history_path, "w") as f:
                json.dump(chat_history, f)
            print("Chat history saved successfully")
            
    except Exception as e:
        error_msg = f"Error saving chat history: {str(e)}"
        print(f"ERROR: {error_msg}")
        handle_exception(e)

def delete_message(file_name, index):
    """Delete a specific message from the chat history."""
    try:
        print(f"Deleting message at index {index} from chat history")
        chat_history = load_chat_history(file_name)
        
        if 0 <= index < len(chat_history):
            print(f"Removing message: {chat_history[index]['content'][:50]}...")
            del chat_history[index]
            
            # If we deleted a user message, also delete the assistant's response
            if index < len(chat_history) and chat_history[index]["role"] == "assistant":
                print("Also removing associated assistant response")
                del chat_history[index]
                
            save_chat_history(file_name, chat_history)
            print("Message deleted successfully")
            
        return chat_history
        
    except Exception as e:
        error_msg = f"Error deleting message: {str(e)}"
        print(f"ERROR: {error_msg}")
        handle_exception(e)
        return load_chat_history(file_name)