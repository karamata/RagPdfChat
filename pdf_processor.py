import PyPDF2
import io
from typing import List, Optional
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Handles PDF text extraction and document processing
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """
        Extract text from PDF file content
        
        Args:
            file_content: PDF file as bytes
            
        Returns:
            str: Extracted text
        """
        try:
            # Create a BytesIO object from file content
            pdf_file = io.BytesIO(file_content)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n"
            
            # Clean the text
            text = self._clean_text(text)
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the current chunk
            end = start + self.chunk_size
            
            # If we're not at the end of the text, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence boundaries within the last 100 characters
                last_period = text.rfind('.', end - 100, end)
                last_exclamation = text.rfind('!', end - 100, end)
                last_question = text.rfind('?', end - 100, end)
                
                # Find the latest sentence boundary
                sentence_end = max(last_period, last_exclamation, last_question)
                
                if sentence_end > start:
                    end = sentence_end + 1
            
            # Extract the chunk
            chunk = text[start:end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move to the next chunk with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= end:
                start = end
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def validate_pdf(self, file_content: bytes) -> bool:
        """
        Validate if the file is a valid PDF
        
        Args:
            file_content: File content as bytes
            
        Returns:
            bool: True if valid PDF
        """
        try:
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Try to access the first page to validate
            if len(pdf_reader.pages) > 0:
                _ = pdf_reader.pages[0]
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False
