PDF Question Answering App
Setup and Installation

Clone the repository
Create a virtual environment:

bashCopypython -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

Install dependencies:

bashCopypip install -r requirements.txt

Run the Streamlit app:

bashCopystreamlit run app.py
Features

PDF upload and processing
Automatic text, table, and image extraction
Semantic search and question answering
Image summarization

Requirements

Python 3.8+
Streamlit
Transformers
Sentence Transformers
PyTorch
