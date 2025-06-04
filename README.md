# RagPdfChat

A simple Retrieval-Augmented Generation (RAG) chat application in Python.  
- **UI:** Streamlit (easy to use, web-based)  
- **LLM:** [Mistral API](https://docs.mistral.ai/)  
- **Embeddings Database:** [Pinecone](https://www.pinecone.io/)  
- **No authentication:** The app is fully public.

## Features

- Upload a PDF file and chat with its content.
- User questions are answered by combining retrieval from PDF context (using embeddings) and a powerful LLM (Mistral).
- Instant, interactive web UI with Streamlit.

## Requirements

- Python 3.8+
- Streamlit
- pinecone-client
- PyPDF2
- requests

Install dependencies:
```bash
pip install streamlit pinecone-client PyPDF2 requests
```

## Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/karamata/RagPdfChat.git
   cd RagPdfChat
   ```

2. **Set API Keys**  
   Your API keys are hardcoded in the code for this demo (no auth required):

   - Mistral API Key:  
     `KjRVvzvLrjWnIGVwFuWt3i3iig5gvyNY`
   - Pinecone API Key:  
     `pcsk_2s3Nn6_5HvwZFers9bidgr5ikkqVhbUC32tjyRi5UzyvvinjKgBdtnzKtnAXo3wVXKcAWe`

   **(You may also edit the `.env` or config in the code to change these keys.)**

## How It Works

1. **Upload PDF:**  
   Users upload a PDF. The app splits the document into chunks, generates embeddings with Mistral, and stores them in Pinecone.

2. **Ask a Question:**  
   - The user types a question.
   - The app retrieves relevant PDF chunks from Pinecone using the question’s embedding.
   - The retrieved context and the question are sent to the Mistral LLM for answer generation.

3. **Get Answer:**  
   - The LLM returns an answer, which is displayed in the chat UI.

## Running the app

```bash
streamlit run app.py
```

Then, open [http://localhost:8501](http://localhost:8501) in your browser.

## Sample `app.py` Structure

```python
import streamlit as st
import pinecone
import requests
from PyPDF2 import PdfReader

MISTRAL_API_KEY = "KjRVvzvLrjWnIGVwFuWt3i3iig5gvyNY"
PINECONE_API_KEY = "pcsk_2s3Nn6_5HvwZFers9bidgr5ikkqVhbUC32tjyRi5UzyvvinjKgBdtnzKtnAXo3wVXKcAWe"
PINECONE_INDEX = "rag-pdf-chat"

# 1. Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
if PINECONE_INDEX not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX, dimension=768)
index = pinecone.Index(PINECONE_INDEX)

# 2. UI to upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file:
    # Extract text and chunking logic here

# 3. UI to ask question
question = st.text_input("Ask a question about your PDF")
if st.button("Ask"):
    # 1. Embed question, search Pinecone for similar chunks
    # 2. Send context + question to Mistral LLM via API
    # 3. Display answer
```

**Note:**  
- This is a simplified example. You’ll need to implement text chunking, embedding, and query logic.
- No authentication is added, the app is fully open.

## License

MIT

---

**Contributions welcome!**  
If you have questions or suggestions, open an issue or pull request.

