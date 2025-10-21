"""
Main Query Engine - orchestrates CSV and PDF queries
"""
import os
from typing import Dict, Any
import pandas as pd
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from src.pdf_handler import PDFHandler
from src.vector_store import VectorStoreManager
from src.helper import (
    create_question_answer_from_context_chain,
    answer_question_from_context
)
from sentence_transformers import SentenceTransformer, util
import sqlite3
from sqlalchemy import create_engine
#from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

class QueryEngine:
    """Main engine to handle all queries"""
    
    def __init__(self, use_gemini: bool = True):
        """
        Initialize query engine
         
        Args:
            use_gemini: Use Gemini (FREE) if True, else HuggingFace
        """
        self.use_gemini = use_gemini

        self.llm = self._get_llm(self.use_gemini)

        if self.llm: 
            self.qa_chain = create_question_answer_from_context_chain(self.llm)
        else:
            self.qa_chain = None
            print("No llm model found")

        # initialize embedding model for query classification
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


    def pdf_init(self):
        #--- qa chat llm init
        self.pdf_handler = None

        self.vector_store = VectorStoreManager(use_gemini=self.use_gemini)
        #self._load_pdfs_to_vectorstore()

        self.vectorstore_loaded = self.vector_store.load_vectorstore(name="pdf_vectorstore")
      
        if self.vectorstore_loaded:
            #print("Existing vectorstore loaded")
            if self.vector_store.vectorstore:
                print("Vectorstore is initialized and ready")
            else:
                print("WARNING: vectorstore_loaded=True but vectorstore is None!")
                self.vectorstore_loaded = False


    def csv_init(self, db_path="local_data.db"):
        #--- sql agent
        # Only create DB if it doesn't exist
        self.db_path = db_path
        self.sql_agent = None
        self.sql_db = None

        if not os.path.exists(self.db_path):
            print("Database not found — creating and loading CSVs...")
            self.db_uri = self.load_csv_to_sqlite()
        else:
            print("Using existing SQLite database.")
            self.db_uri = f"sqlite:///{self.db_path}"

        # Initialize SQL Database + Agent
        self.sql_db = SQLDatabase.from_uri(self.db_uri)
        self.create_sql_agent()

    
    def _get_llm(self, use_gemini: bool):
        """Get LLM (FREE options)"""
        if os.getenv("GOOGLE_API_KEY"):
            return ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
        elif os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            return HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature": 0.3})
        else:
            print("No API key found — using local-only mode (no LLM).")
            return None
    
    def load_csv_to_sqlite(self):
        conn = sqlite3.connect("local_data.db")
        csv_files = ["data/csv/co_sales.csv", "data/csv/industry_sales.csv"]
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            table_name = csv_file.split("/")[-1].replace(".csv", "")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            print(f"{table_name} uploaded to SQLite.")
        conn.close()
        return "sqlite:///local_data.db"


    def create_sql_agent(self):
        """Initialize SQL agent using Gemini LLM."""
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
            self.sql_agent = create_sql_agent(
                llm=llm,
                db=self.sql_db,
                verbose=True,
            )
            print("SQL Agent initialized successfully.")
        except Exception as e:
            print(f"Error creating SQL agent: {e}")


    def query_csv(self, question: str):
        """Use SQL agent to answer natural language questions on CSV tables in Postgres"""
        self.csv_init()

        if not self.sql_agent:
            return {"success": False, "error": "LLM SQL agent not initialized."}

        try:
            result = self.sql_agent.invoke(question)
            return {"success": True, "source": "csv_sql", "answer": result}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _ensure_pdf_loaded(self):
        if self.vectorstore_loaded and self.vector_store.vectorstore:
            print("vector db loaded")
            return
        
        print("Vectorstore not loaded. Initializing PDF loading")
        
        if self.pdf_handler is None:
            print("initializing pdf handler")
            self.pdf_handler = PDFHandler()

        documents = self.pdf_handler.load_all_pdfs()
        if not documents:
            print("No PDFs found to load into vectorstore")
            return 

        chunks = self.pdf_handler.chunk_documents(chunk_size=1000, chunk_overlap=200)
        if not chunks:
            print("No document chunks created for vectorstore")
            return 

        self.vector_store.create_vectorstore(chunks)
        self.vector_store.save_vectorstore()
        print(f"Vectorstore created with {len(chunks)} chunks and saved to disk.")
        self.vectorstore_loaded = True

    
    def query_pdf(self, question: str, k: int = 5) -> Dict[str, Any]:

        # lazy load pdfs if vectorstore is missing
        self.pdf_init()
        self._ensure_pdf_loaded()

        if not self.vector_store.vectorstore:
            return {"success": False, "error": "Vector store not initialized."}

        retriever = self.vector_store.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)

        if not docs:
            return {"success": False, "error": "No relevant PDF content found."}

        # find context
        context = "\n\n".join([d.page_content for d in docs])

        if self.qa_chain:
            answer = answer_question_from_context(question, context, self.qa_chain)
            if answer is None:
                return {
                "success": True,
                "source": "pdf",
                "answer": "LLM failed tp provide an answer - showing relavtn snippets.",
                "sources": [{"filename": d.metadata.get("source", ""), "content": d.page_content} for d in docs]
            }
            return {
            "success": True,
            "source": "pdf",
            "answer": answer["answer"],
            "sources": [{"filename": d.metadata.get("source", ""), "content": d.page_content} for d in docs]
            }

        
        else:
            return {
                "success": True,
                "source": "pdf",
                "answer": "LLM unavailable — showing relevant snippets.",
                "sources": [{"filename": d.metadata.get("source", ""), "content": d.page_content} for d in docs]
            }

    def classify_query(self, question: str) -> str:
        """Classify if query is about CSV data or PDF documents"""
        if not question or len(question.strip()) == 0:
            return "unknown"

        question_lower = question.lower()

        # Define keyword anchors for classification
        #self.csv_keywords = ["sales", "data", "revenue", "vehicle", "count", "profit", "date"]
        #self.pdf_keywords = ["report", "strategy", "market", "funding", "industry", "summary", "document"]

        self.csv_keywords = ["average", "sum", "trend", "count", "growth", "percentage", "compare", "table", "correlation", "dataset", "month", "date", "total", "revenue", "sales", "query", "group by", "filter"]
        self.pdf_keywords = ["summary", "explain", "describe", "insight", "conclusion", "strategy", "market", "industry", "overview", "objective",  "policy", "goal", "purpose", "highlight"]

        # Quick keyword count 
        csv_score_kw = sum(kw in question_lower for kw in self.csv_keywords)
        pdf_score_kw = sum(kw in question_lower for kw in self.pdf_keywords)
        
        # Semantic similarity using embeddings
        q_emb = self.model.encode(question, convert_to_tensor=True)
        csv_embs = self.model.encode(self.csv_keywords, convert_to_tensor = True)
        pdf_embs = self.model.encode(self.pdf_keywords, convert_to_tensor = True)

        csv_sim = float(util.cos_sim(q_emb, csv_embs).max())
        pdf_sim = float(util.cos_sim(q_emb, pdf_embs).max())

        # Weighted final decision
        csv_score = csv_score_kw + csv_sim
        pdf_score = pdf_score_kw + pdf_sim

        return "csv" if csv_score > pdf_score else "pdf"

    
    def process_query(self, question: str) -> Dict[str, Any]:
        """Main method to process any query"""
        query_type = self.classify_query(question)
        print("classified query type:", query_type)
        
        if query_type == 'csv':
            return self.query_csv(question)
        
        else:
            return self.query_pdf(question)
        
    #  def classify_query_llm(self, question: str) -> str:
    #     """Classify query type using LLM"""
    #     if not self.llm:
    #         return self.classify_query(question)  # fallback to keyword+embedding

    #     prompt = f"""You are a classifier. Categorize the user's question as one of:
    #     1 . 'csv' — if the query asks for numbers, counts, comparisons, or data analysis (like a SQL query)
    #     2. 'pdf' — if the query asks for explanations, summaries, or conceptual/theoretical details.
    #     User question: {question}"""

    #     try:
    #         result = self.llm.invoke(prompt)
    #         res_lower = result.content.lower()
    #         if "csv" in res_lower:
    #             return "csv"
    #         elif "pdf" in res_lower:
    #             return "pdf"
    #         else:
    #             return "mixed"

    #     except Exception:
    #         return self.classify_query(question)


