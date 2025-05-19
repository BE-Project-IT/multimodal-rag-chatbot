import uuid
import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document
from src.exception_handler import handle_exception
import streamlit as st

class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Initializing SentenceTransformer with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"SentenceTransformer initialized successfully")
    
    def embed_documents(self, texts):
        print(f"Embedding {len(texts)} documents...")
        embeddings = self.model.encode(texts)
        print(f"Document embedding complete. Shape: {embeddings.shape}")
        return embeddings.tolist()
    
    def embed_query(self, text):
        print(f"Embedding query: {text[:50]}..." if len(text) > 50 else f"Embedding query: {text}")
        embedding = self.model.encode([text])[0]
        print(f"Query embedding complete. Shape: {len(embedding)}")
        return embedding.tolist()

class VectorStoreManager:
    @staticmethod
    def create_vector_store(processed_data, store_name):
        print(f"Creating vector store: {store_name}")
        documents = []
        status = st.empty()
        
        try:
            # Process text elements
            if processed_data['text_elements']:
                status.info(f"Creating vector store: Processing {len(processed_data['text_elements'])} text elements...")
                print(f"Processing {len(processed_data['text_elements'])} text elements...")
                
                for i, (e, s) in enumerate(zip(processed_data['text_elements'], processed_data['text_summaries'])):
                    print(f"Text element {i+1}: ID={str(uuid.uuid4())[:8]}, Summary={s[:50]}...")
                    doc_id = str(uuid.uuid4())
                    doc = Document(
                        page_content=s,
                        metadata={'id': doc_id, 'type': 'text', 'original_content': e}
                    )
                    documents.append(doc)
            
            # Process table elements
            if processed_data['table_elements']:
                status.info(f"Creating vector store: Processing {len(processed_data['table_elements'])} table elements...")
                print(f"Processing {len(processed_data['table_elements'])} table elements...")
                
                for i, (e, s) in enumerate(zip(processed_data['table_elements'], processed_data['table_summaries'])):
                    print(f"Table element {i+1}: ID={str(uuid.uuid4())[:8]}, Summary={s[:50]}...")
                    doc_id = str(uuid.uuid4())
                    doc = Document(
                        page_content=s,
                        metadata={'id': doc_id, 'type': 'table', 'original_content': e}
                    )
                    documents.append(doc)
            
            # Process image elements
            if processed_data['image_paths']:
                status.info(f"Creating vector store: Processing {len(processed_data['image_paths'])} image elements...")
                print(f"Processing {len(processed_data['image_paths'])} image elements...")
                
                for i, (e, s) in enumerate(zip(processed_data['image_paths'], processed_data['image_summaries'])):
                    print(f"Image element {i+1}: Path={e}, Summary={s[:50]}...")
                    doc_id = str(uuid.uuid4())
                    doc = Document(
                        page_content=s,
                        metadata={'id': doc_id, 'type': 'image', 'original_content': e}
                    )
                    documents.append(doc)
            
            # Create embeddings and vector store
            status.info(f"Creating vector store: Generating embeddings for {len(documents)} elements...")
            print(f"Initializing embeddings model...")
            embeddings = LocalSentenceTransformerEmbeddings()
            
            print(f"Creating FAISS vector store from {len(documents)} documents...")
            vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
            
            # Save vector store
            print(f"Creating directory for vector store: {store_name}")
            os.makedirs(store_name, exist_ok=True)
            
            print(f"Saving vector store to {store_name}...")
            vectorstore.save_local(store_name)
            
            success_msg = f"Vector store '{store_name}' created with {len(documents)} documents."
            print(success_msg)
            status.success(success_msg)
            
            return vectorstore, embeddings
        except Exception as e:
            error_msg = f"Error creating vector store: {str(e)}"
            print(f"ERROR: {error_msg}")
            handle_exception(e)
            raise

    @staticmethod
    def get_embeddings():
        print("Creating new embedding model instance...")
        return LocalSentenceTransformerEmbeddings()

    @staticmethod
    def load_vector_store(embeddings, store_name):
        try:
            print(f"Loading vector store from {store_name}...")
            status = st.empty()
            status.info(f"Loading vector store: {store_name}...")
            
            vectorstore = FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)
            
            print(f"Vector store loaded successfully: {store_name}")
            status.success(f"Vector store loaded successfully: {store_name}")
            
            return vectorstore
        except Exception as e:
            error_msg = f"Error loading vector store {store_name}: {str(e)}"
            print(f"ERROR: {error_msg}")
            handle_exception(e)
            raise