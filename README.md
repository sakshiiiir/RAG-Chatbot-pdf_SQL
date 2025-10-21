### Chatbot EV — RAG Chatbot for EV Data

This project implements a Retrieval-Augmented Generation (RAG) Chatbot for electric vehicle (EV) data. The chatbot can handle structured CSV data and unstructured PDF documents, allowing users to ask questions and get answers via a Streamlit interface.

Data Used:
PDFs: Sales and annual reports (uploaded test file here instead of the actual pdfs)
CSV Files: Company sales and industry data (uploaded test file here instead of the actual pdfs)

Folder structure:
- app.py: Streamlit front-end
- tester.py: Python script to test queries locally
-  src/
  - query_engine.py   # Main query engine handling CSV and PDF queries
  - pdf_handler.py    # Load PDFs → convert to documents → chunk → embeddings
  - helper.py         # LLM helper functions (QA from context, embeddings)
  - vector_store.py   # FAISS vector database management
- data
  - csv: co sales.csv, industry sales.csv
  - pdfs - subfolders
  - cache/faiss_index/pdf_vectorstore/ ; where embeddings faiss are extracted and stored.

How It Works
1. User Input Flow
User enters a question in the Streamlit UI. The question is converted to embedding.
Based on classification (PDF or CSV), it is routed to:
- PDF Query: Unstructured data search 
- CSV Query: Structured SQL query

2. Classification
Keyword-based and embedding similarity comparison. (currently using this)
LLM-based classification is optional but avoided for simplicity.

3. PDF Query (Unstructured Data)
Vector Database: FAISS
Process:
- Load PDFs → Convert to text → Chunk → Create embeddings
- Search relevant chunks using question embeddings
- Use LLM QA chain to answer from context

4. LLM Functions Used:
- Embeddings creation
- Document chunking
- QA chain initialization
- Answer generation from context

5. CSV Query (Structured Data)
SQL Agent: Queries CSV data stored in SQLite.
LLM: Gemini 2.0 Flash (via Google Generative AI)
Process:
- CSVs loaded into SQLite database
- Gemini SQL agent generates SQL queries based on user question
- Query results are returned

6. System Architecture
   
<img width="992" height="1500" alt="image" src="https://github.com/user-attachments/assets/1cd66b81-c307-4e6e-8c46-9f4ec55c0b11" />


7. Technologies Used:
- LLM: Gemini API (QnA), gemini-2.5-pro(for qna), gemini-2.0-flash (for sql agent)
- Vector Database: FAISS
- Embeddings: HuggingFace SentenceTransformer (all-MiniLM-L6-v2) or Gemini
- UI: Streamlit
- Database: SQLite for structured CSV data
  
8. how to run:
  Give API Keys in .env file
  python -m venv env && source env/bin/activate for environment setup.
  pip install -r requirements.txt
  streamlit run app.py
  python tester.py

  streamlit images:
  user input: Tell me about Ather's recent strategy
<img width="1710" height="1093" alt="chatbot-qna-pdf" src="https://github.com/user-attachments/assets/36b879f3-3c73-4295-bb5f-43ae750c442d" />

  user input: What are the top selling models last quarter
<img width="2024" height="1470" alt="image" src="https://github.com/user-attachments/assets/36fd71b5-3725-4ce7-b995-9dba7194df03" />



