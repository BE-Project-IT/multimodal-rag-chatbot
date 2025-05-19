import os
import base64
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.exception_handler import handle_exception
import streamlit as st

class QuestionAnswerer:
    def __init__(self, model_name="Qwen/Qwen1.5-1.8B-Chat"):
        try:
            print(f"Initializing QuestionAnswerer with model: {model_name}")
            status = st.empty()
            status.info(f"Loading language model: {model_name}...")
            
            print(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Tokenizer loaded successfully")
            
            print(f"Loading LLM model {model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.device = self.model.device
            print(f"Model loaded successfully on device: {self.device}")
            
            status.success(f"Question answering system initialized successfully")
            status.empty()
            
        except Exception as e:
            error_msg = f"Error initializing QuestionAnswerer: {str(e)}"
            print(f"ERROR: {error_msg}")
            handle_exception(e)
            raise

    def answer_question(self, vectorstore, question):
        try:
            print(f"Processing question: '{question}'")
            status = st.empty()
            status.info("Finding relevant information...")
            
            # Retrieve relevant documents
            print("Retrieving relevant documents from vector store...")
            relevant_docs = vectorstore.similarity_search(question)
            print(f"Retrieved {len(relevant_docs)} relevant documents")
            
            # Process documents to extract context and images
            print("Processing retrieved documents...")
            context_parts, relevant_images = self._process_documents(relevant_docs)
            print(f"Processed {len(context_parts)} context parts and {len(relevant_images)} relevant images")
            
            # Log contexts
            print("Context parts for answering:")
            for i, ctx in enumerate(context_parts):
                print(f"Context {i+1}: {ctx[:100]}...")
            
            # Create prompt for LLM
            status.info("Formulating response...")
            print("Creating prompt for model...")
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions based on the provided context ONLY."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context: {' '.join(context_parts)}\nAnswer this question: {question}"
                }
            ]

            # Format prompt
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print(f"Formatted prompt: {text[:200]}...")

            # Encode and generate response
            print("Encoding prompt and generating response...")
            inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            print("Input shape:", inputs["input_ids"].shape)
            print("Generating response with temperature=0.5, max_new_tokens=700...")
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=700,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )

            # Extract and clean response
            print("Extracting generated text...")
            gen_ids = [
                out[len(inp):]
                for inp, out in zip(inputs["input_ids"], outputs)
            ]
            response = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
            print(f"Generated response: {response[:200]}...")

            return response, relevant_images

        except Exception as e:
            error_msg = f"Error answering question: {str(e)}"
            print(f"ERROR: {error_msg}")
            handle_exception(e)
            raise

    def _process_documents(self, relevant_docs):
        print(f"Processing {len(relevant_docs)} relevant documents")
        context_parts, relevant_images = [], []

        # Process image documents
        print("Processing image documents...")
        image_docs = [d for d in relevant_docs if d.metadata.get("type") == "image"]
        print(f"Found {len(image_docs)} image documents")
        
        for i, img in enumerate(image_docs):
            path = os.path.join("figures", img.metadata["original_content"])
            print(f"Processing image {i+1}: {path}")
            
            if os.path.exists(path):
                print(f"Image file exists, reading and encoding...")
                with open(path, "rb") as f:
                    img_bytes = f.read()
                    encoded = base64.b64encode(img_bytes).decode()
                    relevant_images.append(encoded)
                    print(f"Image {i+1} encoded successfully, size: {len(encoded)} bytes")
            else:
                print(f"WARNING: Image file not found: {path}")

        # Assemble context from all document types
        print("Assembling context parts from all documents...")
        for i, doc in enumerate(relevant_docs):
            doc_type = doc.metadata.get("type", "unknown")
            print(f"Document {i+1} type: {doc_type}")
            
            prefix = {
                "text": "[text]",
                "table": "[table]",
                "image": "[image]"
            }.get(doc_type, "[unk]")
            
            if doc_type != "image":
                content = doc.metadata.get("original_content")
                print(f"Content excerpt: {content[:100]}...")
            else:
                content = doc.page_content
                print(f"Image description: {content[:100]}...")
                
            context_part = f"{prefix}{content}"
            context_parts.append(context_part)
            print(f"Added context part {i+1} with prefix {prefix}")

        print(f"Assembled {len(context_parts)} context parts")
        return context_parts, relevant_images