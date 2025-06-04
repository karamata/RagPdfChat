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
    
    def extract_text_from_document(self, file_content: bytes, filename: str) -> str:
        """
        Extract text from document file content
        
        Args:
            file_content: Document file as bytes
            filename: Name of the file
            
        Returns:
            str: Extracted text
        """
        try:
            # Check file type
            if filename.lower().endswith('.txt'):
                # Handle text files
                try:
                    text = file_content.decode('utf-8')
                    logger.info(f"Extracted {len(text)} characters from text file")
                    return self._clean_text(text)
                except UnicodeDecodeError:
                    try:
                        text = file_content.decode('latin-1')
                        logger.info(f"Extracted {len(text)} characters from text file (latin-1)")
                        return self._clean_text(text)
                    except Exception as e:
                        logger.error(f"Failed to decode text file: {str(e)}")
                        raise Exception(f"Failed to decode text file: {str(e)}")
            
            elif filename.lower().endswith('.pdf'):
                # Handle PDF files
                try:
                    import PyPDF2
                    
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
                    
                except ImportError:
                    # Basic PDF text extraction fallback
                    logger.warning("PyPDF2 not available, using basic PDF text extraction")
                    
                    # Try to extract basic text patterns from PDF
                    content_str = file_content.decode('latin-1', errors='ignore')
                    
                    # Look for text between stream objects and other text patterns
                    import re
                    
                    # Multiple patterns for different PDF text encodings
                    patterns = [
                        r'BT\s*(.*?)\s*ET',  # Text objects
                        r'\((.*?)\)',        # Text in parentheses
                        r'<(.*?)>',          # Text in angle brackets
                    ]
                    
                    extracted_text = ""
                    for pattern in patterns:
                        matches = re.findall(pattern, content_str, re.DOTALL)
                        for match in matches:
                            # Clean up PDF text commands
                            clean_match = re.sub(r'/\w+\s*', '', match)
                            clean_match = re.sub(r'Tf\s*', '', clean_match)
                            clean_match = re.sub(r'Td\s*', ' ', clean_match)
                            clean_match = re.sub(r'Tj\s*', '', clean_match)
                            clean_match = re.sub(r'\d+\.?\d*\s*', '', clean_match)
                            clean_match = re.sub(r'[()<>]', '', clean_match)
                            if len(clean_match.strip()) > 2:  # Only add meaningful text
                                extracted_text += clean_match + " "
                    
                    if extracted_text.strip():
                        text = self._clean_text(extracted_text)
                        logger.info(f"Extracted {len(text)} characters using basic PDF parsing")
                        return text
                    else:
                        raise Exception("Could not extract text from PDF. Please ensure the PDF contains readable text or install PyPDF2 for better extraction.")
            
            else:
                raise Exception(f"Unsupported file type: {filename}")
            
        except Exception as e:
            logger.error(f"Error extracting text from document: {str(e)}")
            raise Exception(f"Failed to extract text from document: {str(e)}")
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """
        Legacy method for backward compatibility
        """
        return self.extract_text_from_document(file_content, "document.pdf")
    
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
            # Basic PDF header validation
            if file_content.startswith(b'%PDF-'):
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"PDF validation failed: {str(e)}")
            return False
