"""
FAISS Vector Store Manager - supports multiple FREE embedding options
"""
import os
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class VectorStoreManager:
    """Manage FAISS vector store with FREE embeddings"""
    
    def __init__(self, use_gemini: bool = True):
      
        self.cache_dir = Path("cache/faiss_index")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore = None
        self.use_gemini = False
        self.embeddings = self._get_embeddings()
    
    def _get_embeddings(self):
        """Get embedding model (FREE options)"""
        # if self.use_gemini and os.getenv("GOOGLE_API_KEY"):
        #     print("🔹 Using Gemini embeddings (FREE)")
        #     return GoogleGenerativeAIEmbeddings(
        #         model="models/embedding-001",
        #         google_api_key=os.getenv("GOOGLE_API_KEY")
        #     )
        # else:
        print("🔹 Using HuggingFace embeddings (FREE, runs locally)")
        # This runs on your computer, no API key needed
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    
    def create_vectorstore(self, documents: List):
        """Create FAISS vector store from documents"""
        if not documents:
            print("No documents to process")
            return False
        
        print(f"Creating vector store with {len(documents)} chunks...")
        
        try:
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            print("✅ Vector store created!")
            return True
        
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False
    
    def save_vectorstore(self, name: str = "pdf_vectorstore"):
        """Save vector store to disk"""
        if not self.vectorstore:
            print("No vector store to save")
            return False
        
        try:
            save_path = self.cache_dir / name
            self.vectorstore.save_local(str(save_path))
            print(f"Vector store saved to: {save_path}")
            return True
        
        except Exception as e:
            print(f"Error saving vector store: {e}")
            return False
    
    def load_vectorstore(self, name: str = "pdf_vectorstore"):
        """Load vector store from disk"""
        load_path = self.cache_dir / name
        
        if not load_path.exists():
            print(f"Vector store not found: {load_path}")
            return False
        
        try:
            self.vectorstore = FAISS.load_local(
                str(load_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Vector store loaded from: {load_path}")
            return True
        
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5):
        """Search for similar documents"""
        if not self.vectorstore:
            print("Vector store not initialized")
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 5):
        """Search with relevance scores"""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score)
                }
                for doc, score in results
            ]
        
        except Exception as e:
            print(f"Error in search: {e}")
            return []
    
    def get_retriever(self, search_kwargs: dict = None):
        """Get LangChain retriever for the vector store"""
        if not self.vectorstore:
            return None
        
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)