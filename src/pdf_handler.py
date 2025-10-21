"""
Handle PDF operations using existing helper functions
"""
from pathlib import Path
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.helper import replace_t_with_space, read_pdf_to_string

class PDFHandler:
    """Handle PDF document processing"""
    
    def __init__(self, data_dir: str= "data/pdfs"):
        self.data_dir = Path(data_dir)
        self.annual_reports_path = self.data_dir / "Annual Reports"
        self.industry_reports_path = self.data_dir / "Industry Reports"
        self.documents = []
        self.chunked_documents = []
    
    def load_all_pdfs(self):
        """Load all PDFs from both directories"""
        self.documents = []
        
        # Load annual reports
        if self.annual_reports_path.exists():
            annual_pdfs = list(self.annual_reports_path.rglob("*.pdf"))
            print(f"Found {len(annual_pdfs)} annual reports")
            # for pdf in annual_pdfs:
                # self._load_single_pdf(pdf, "annual_report")
        
        # Load industry reports
        if self.industry_reports_path.exists():
            industry_pdfs = list(self.industry_reports_path.rglob("*.pdf"))
            print(f"Found {len(industry_pdfs)} industry reports")
            # for pdf in industry_pdfs:
            #     self._load_single_pdf(pdf, "industry_report")
        
        print(f"Total PDFs loaded: {len(self.documents)}")
        return self.documents
    
    def _load_single_pdf(self, pdf_path: Path, doc_type: str):
        """Load a single PDF using LangChain"""
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # Add metadata
            for page in pages:
                page.metadata['source_file'] = pdf_path.name
                page.metadata['doc_type'] = doc_type
                page.metadata['full_path'] = str(pdf_path)
            
            self.documents.extend(pages)
            print(f" Loaded: {pdf_path.name} ({len(pages)} pages)")
        
        except Exception as e:
            print(f"Error loading {pdf_path.name}: {e}")
    
    def chunk_documents(self, chunk_size: int = 1000, 
                       chunk_overlap: int = 200) -> List:
        """Chunk documents for embedding"""
        if not self.documents:
            print("No documents to chunk")
            return []
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Split and clean
        chunks = text_splitter.split_documents(self.documents)
        self.chunked_documents = replace_t_with_space(chunks)
        
        print(f"Created {len(self.chunked_documents)} chunks")
        return self.chunked_documents
    
    def get_document_summary(self) -> Dict:
        """Get summary of loaded documents"""
        annual_count = sum(1 for doc in self.documents 
                          if doc.metadata.get('doc_type') == 'annual_report')
        industry_count = sum(1 for doc in self.documents 
                            if doc.metadata.get('doc_type') == 'industry_report')
        
        return {
            'total_documents': len(self.documents),
            'annual_reports': annual_count,
            'industry_reports': industry_count,
            'total_chunks': len(self.chunked_documents),
            'unique_files': len(set(doc.metadata.get('source_file') 
                                   for doc in self.documents))
        }
    
    def search_by_filename(self, filename: str) -> List:
        """Get all chunks from a specific file"""
        return [doc for doc in self.chunked_documents 
                if doc.metadata.get('source_file') == filename]


pdf = PDFHandler()
pdf.load_all_pdfs()