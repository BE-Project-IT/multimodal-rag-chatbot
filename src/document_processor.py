import os
from PIL import Image
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.doc import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.image import partition_image
from src.exception_handler import handle_exception
import streamlit as st

class DocumentProcessor:
    def __init__(self, output_path="figures"):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        print(f"DocumentProcessor initialized with output path: {self.output_path}")
        
        print("Loading summarization model (facebook/bart-large-cnn)...")
        device_index = 0 if torch.cuda.is_available() else -1
        print(f"Using device index: {device_index} ({torch.device('cuda' if torch.cuda.is_available() else 'cpu')})")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device_index)
        print("Summarization model loaded successfully")
        
        print("Loading image model (vikhyatk/moondream2)...")
        self.image_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            revision="2024-08-26",
            torch_dtype=torch.float16
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Image model loaded successfully")
        
        self.image_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-08-26")
        print("Image tokenizer loaded successfully")

    def process(self, filename: str, filetype: str):
        print(f"Processing file {filename} of type {filetype}...")
        status = st.empty()
        status.info(f"Processing {filetype.upper()} document: Extracting content...")
        
        try:
            if filetype == "pdf":
                print(f"Using PDF extraction strategy: auto, extracting images and tables...")
                status.info(f"Processing PDF: Extracting text, images, and tables...")
                elements = partition_pdf(
                    filename=filename,
                    strategy='auto',
                    extract_images_in_pdf=True,
                    extract_image_block_types=["Image", "Table"],
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                    max_characters=4000,
                    new_after_n_chars=3800,
                    combine_text_under_n_chars=2000,
                    image_output_dir_path=self.output_path,
                )
                print(f"PDF extraction complete. Found {len(elements)} elements")
                
            elif filetype == "docx":
                print(f"Extracting content from DOCX file...")
                status.info(f"Processing DOCX: Extracting document content...")
                elements = partition_docx(filename=filename)
                print(f"DOCX extraction complete. Found {len(elements)} elements")
                
            elif filetype == "pptx":
                print(f"Extracting content from PPTX file...")
                status.info(f"Processing PPTX: Extracting slides content...")
                elements = partition_pptx(filename=filename)
                print(f"PPTX extraction complete. Found {len(elements)} elements")
                
            elif filetype == "txt":
                print(f"Extracting content from TXT file...")
                status.info(f"Processing TXT: Extracting text content...")
                elements = partition_text(filename=filename)
                print(f"TXT extraction complete. Found {len(elements)} elements")
                
            elif filetype == "xlsx":
                print(f"Extracting content from XLSX file...")
                status.info(f"Processing XLSX: Extracting spreadsheet content...")
                elements = partition_xlsx(filename=filename)
                print(f"XLSX extraction complete. Found {len(elements)} elements")
                
            elif filetype in ["jpg", "jpeg", "png", "heic"]:
                print(f"Extracting content from image file using auto strategy...")
                status.info(f"Processing image: Extracting visual content...")
                elements = partition_image(filename=filename, strategy="auto", languages=["eng"])
                print(f"Image extraction complete. Found {len(elements)} elements")
                
            else:
                error_msg = f"Unsupported file type: {filetype}"
                print(f"ERROR: {error_msg}")
                status.error(error_msg)
                raise ValueError(error_msg)
                
            status.info(f"Processing {filetype.upper()}: Analyzing extracted content...")
            print("Beginning element processing and summarization...")
            processed = self._process_elements(elements)
            print("Processing complete. Summary of results:")
            print(f"- Text elements: {len(processed['text_elements'])}")
            print(f"- Table elements: {len(processed['table_elements'])}")
            print(f"- Image paths: {len(processed['image_paths'])}")
            print(f"- Text summaries: {len(processed['text_summaries'])}")
            print(f"- Table summaries: {len(processed['table_summaries'])}")
            print(f"- Image summaries: {len(processed['image_summaries'])}")
            status.success(f"Document processing complete: Found {len(processed['text_elements'])} text blocks, {len(processed['table_elements'])} tables, and {len(processed['image_paths'])} images")
            
            return processed
        except Exception as e:
            error_msg = f"Error processing file {filename}: {str(e)}"
            print(f"ERROR: {error_msg}")
            status.error(error_msg)
            handle_exception(e)
            raise

    def _process_elements(self, raw_elements):
        print("Processing extracted elements...")
        text_elements, table_elements, image_paths = [], [], []
        
        # Process raw elements
        print("Categorizing elements by type...")
        for i, el in enumerate(raw_elements):
            el_type = str(type(el))
            print(f"Element {i}: Type = {el_type}")
            
            if "CompositeElement" in el_type:
                print(f"Found text element: {el.text[:50]}..." if len(el.text) > 50 else f"Found text element: {el.text}")
                text_elements.append(el.text)
            elif "Table" in el_type:
                print(f"Found table element: {el.text[:50]}..." if len(el.text) > 50 else f"Found table element: {el.text}")
                table_elements.append(el.text)
        
        # Find all image files in output directory
        print(f"Scanning {self.output_path} directory for extracted images...")
        for file in os.listdir(self.output_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".heic")):
                print(f"Found image file: {file}")
                image_paths.append(file)
        
        # Create initial structure
        result = {
            "text_elements": text_elements,
            "table_elements": table_elements,
            "image_paths": image_paths,
        }
        
        # Process text elements
        status = st.empty()
        if text_elements:
            status.info(f"Summarizing {len(text_elements)} text blocks...")
            print(f"Summarizing {len(text_elements)} text elements...")
            result["text_summaries"] = self._summarize_texts(text_elements)
        else:
            print("No text elements to summarize")
            result["text_summaries"] = []
            
        # Process table elements
        if table_elements:
            status.info(f"Summarizing {len(table_elements)} tables...")
            print(f"Summarizing {len(table_elements)} table elements...")
            result["table_summaries"] = self._summarize_tables(table_elements)
        else:
            print("No table elements to summarize")
            result["table_summaries"] = []
            
        # Process image elements
        if image_paths:
            status.info(f"Analyzing {len(image_paths)} images...")
            print(f"Summarizing {len(image_paths)} images...")
            result["image_summaries"] = self._summarize_images(image_paths)
        else:
            print("No image elements to summarize")
            result["image_summaries"] = []
            
        return result

    def _summarize_texts(self, texts):
        summaries = []
        print(f"Starting text summarization for {len(texts)} text blocks")
        status = st.empty()
        
        for i, text in enumerate(texts):
            try:
                status.info(f"Summarizing text {i+1}/{len(texts)}...")
                print(f"Summarizing text {i+1}/{len(texts)}: {text[:100]}...")
                
                summary = self.summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                print(f"Summary: {summary}")
                summaries.append(summary)
                
            except Exception as e:
                error_msg = f"Text summarization error for text {i+1}: {str(e)}"
                print(f"ERROR: {error_msg}")
                summaries.append("Summary not available.")
        
        print(f"Text summarization complete. Generated {len(summaries)} summaries.")
        return summaries

    def _summarize_tables(self, tables):
        summaries = []
        print(f"Starting table summarization for {len(tables)} tables")
        status = st.empty()
        
        for i, table in enumerate(tables):
            try:
                status.info(f"Summarizing table {i+1}/{len(tables)}...")
                print(f"Summarizing table {i+1}/{len(tables)}: {table[:100]}...")
                
                summary = self.summarizer(table)[0]['summary_text']
                print(f"Table summary: {summary}")
                summaries.append(summary)
                
            except Exception as e:
                error_msg = f"Table summarization error for table {i+1}: {str(e)}"
                print(f"ERROR: {error_msg}")
                summaries.append("Summary not available.")
        
        print(f"Table summarization complete. Generated {len(summaries)} summaries.")
        return summaries

    def _summarize_images(self, image_paths):
        summaries = []
        print(f"Starting image summarization for {len(image_paths)} images")
        status = st.empty()
        
        for i, img_name in enumerate(image_paths):
            try:
                status.info(f"Analyzing image {i+1}/{len(image_paths)}: {img_name}")
                print(f"Summarizing image {i+1}/{len(image_paths)}: {img_name}")
                
                img_path = os.path.join(self.output_path, img_name)
                print(f"Loading image from: {img_path}")
                img = Image.open(img_path)
                
                print("Encoding image for moondream2 model...")
                enc_img = self.image_model.encode_image(img)
                
                print("Generating image description...")
                summary = self.image_model.answer_question(enc_img, "Describe this image in detail", self.image_tokenizer)
                print(f"Image summary: {summary}")
                summaries.append(summary)
                
            except Exception as e:
                error_msg = f"Image summarization failed for {img_name}: {str(e)}"
                print(f"ERROR: {error_msg}")
                summaries.append("Image summary failed.")
        
        print(f"Image summarization complete. Generated {len(summaries)} summaries.")
        return summaries