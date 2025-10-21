import os
from src.query_engine import QueryEngine
from dotenv import load_dotenv
load_dotenv()

def run_local_test():
    print("\n Local Test for RAG Chatbot Engine\n")

    # === 1️Initialize Query Engine ===
    # use_gemini=True → Gemini model via google-generativeai
    # use_gemini=False → HuggingFace pipeline
    engine = QueryEngine(use_gemini=True)


    # # ===  Ask test questions ===
    test_questions = [
        "Tell me about Ather’s recent strategy"
        # What were the top-selling models last quarter?"
        #"How much funding did Ola Electric raise last year?" ,
        #"Which region had the highest sales?"
    ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        try:
            response = engine.process_query(q)
            print("Answer:", response.get("answer", "No response"))
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    run_local_test()

    
# loading vectordb
# from src.vector_store import VectorStoreManager
# from src.pdf_handler import PDFHandler

# def main():
    # # Initialize PDF handler
    # pdf_handler = PDFHandler()

    # # Initialize vector store manager
    # vector_store = VectorStoreManager()

    # # Try loading saved vectorstore first
    # loaded = vector_store.load_vectorstore()
    # if loaded:
    #     print("✅ Loaded existing vectorstore from disk.")
    # else:
    #     print("⚡ No existing vectorstore found. Creating a new one...")

    #     # Load all PDFs
    #     documents = pdf_handler.load_all_pdfs()
    #     if not documents:
    #         print("No PDFs found to process.")
    #         return

    #     # Chunk documents
    #     chunks = pdf_handler.chunk_documents(chunk_size=1000, chunk_overlap=200)
    #     if not chunks:
    #         print("No document chunks created.")
    #         return

    #     # Create vectorstore from chunks
    #     created = vector_store.create_vectorstore(chunks)
    #     if created:
    #         # Save it for future use
    #         vector_store.save_vectorstore()
    #         print(f"Vectorstore created with {len(chunks)} chunks and saved to disk.")

