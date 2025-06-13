"""
PDF Processor Module
Handles PDF file processing and text extraction
"""

import os
import logging
import tempfile
from typing import List, Dict, Tuple
import PyPDF2
import fitz  # PyMuPDF
import io
from PIL import Image
import pytesseract
import cv2
import numpy as np

import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Handles PDF file processing and text extraction using multiple methods
    """
    
    def __init__(self):
        """
        Initialize PDF processor
        """
        try:
            # Set up OCR fallback if needed
            self.ocr_available = self._check_tesseract_availability()
            logger.info("PDF Processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing PDF Processor: {str(e)}")
            raise
    
    def _check_tesseract_availability(self) -> bool:
        """
        Check if Tesseract is available for OCR fallback
        
        Returns:
            bool: True if Tesseract is available
        """
        try:
            if os.path.exists(config.TESSERACT_CMD):
                pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
                return True
            else:
                # Try to find tesseract in PATH
                pytesseract.get_tesseract_version()
                return True
        except:
            logger.warning("Tesseract not found. OCR fallback will not be available.")
            return False
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from PDF file using multiple methods
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict: Contains extracted text, page count, and metadata
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Validate PDF file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # First try PyMuPDF for text extraction
            extracted_data = self._extract_with_pymupdf(pdf_path)
            
            # If text extraction failed or returned minimal text, try PyPDF2
            if not extracted_data['full_text'].strip() or len(extracted_data['full_text'].strip()) < 100:
                logger.info("PyMuPDF extraction insufficient, trying PyPDF2...")
                extracted_data = self._extract_with_pypdf2(pdf_path)
            
            # If still no good text and OCR is available, try OCR as fallback
            if (not extracted_data['full_text'].strip() or 
                len(extracted_data['full_text'].strip()) < 50) and self.ocr_available:
                logger.info("Text extraction insufficient, trying OCR fallback...")
                extracted_data = self._extract_with_ocr_fallback(pdf_path)
            
            logger.info(f"Successfully processed PDF with {extracted_data['page_count']} pages")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text using PyMuPDF (fitz)
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict: Extracted data
        """
        try:
            doc = fitz.open(pdf_path)
            
            extracted_data = {
                'pages': [],
                'full_text': '',
                'page_count': len(doc),
                'file_name': os.path.basename(pdf_path),
                'extraction_method': 'PyMuPDF'
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                page_data = {
                    'page_number': page_num + 1,
                    'text': page_text,
                    'word_count': len(page_text.split())
                }
                
                extracted_data['pages'].append(page_data)
                extracted_data['full_text'] += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            logger.info(f"PyMuPDF extracted {len(extracted_data['full_text'])} characters")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error with PyMuPDF extraction: {str(e)}")
            raise
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text using PyPDF2
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict: Extracted data
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                extracted_data = {
                    'pages': [],
                    'full_text': '',
                    'page_count': len(pdf_reader.pages),
                    'file_name': os.path.basename(pdf_path),
                    'extraction_method': 'PyPDF2'
                }
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    
                    page_data = {
                        'page_number': page_num + 1,
                        'text': page_text,
                        'word_count': len(page_text.split())
                    }
                    
                    extracted_data['pages'].append(page_data)
                    extracted_data['full_text'] += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                logger.info(f"PyPDF2 extracted {len(extracted_data['full_text'])} characters")
                return extracted_data
                
        except Exception as e:
            logger.error(f"Error with PyPDF2 extraction: {str(e)}")
            raise
    
    def _extract_with_ocr_fallback(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text using OCR as fallback method
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict: Extracted data
        """
        try:
            if not self.ocr_available:
                raise Exception("OCR not available")
            
            doc = fitz.open(pdf_path)
            
            extracted_data = {
                'pages': [],
                'full_text': '',
                'page_count': len(doc),
                'file_name': os.path.basename(pdf_path),
                'extraction_method': 'OCR_Fallback'
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # Zoom factor
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(img_data))
                
                # Apply OCR
                page_text = pytesseract.image_to_string(
                    image, 
                    config=config.TESSERACT_CONFIG if hasattr(config, 'TESSERACT_CONFIG') else ''
                )
                
                page_data = {
                    'page_number': page_num + 1,
                    'text': page_text,
                    'word_count': len(page_text.split())
                }
                
                extracted_data['pages'].append(page_data)
                extracted_data['full_text'] += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            doc.close()
            logger.info(f"OCR fallback extracted {len(extracted_data['full_text'])} characters")
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error with OCR fallback: {str(e)}")
            raise
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract metadata from PDF file
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Dict: PDF metadata
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                metadata = {
                    'page_count': len(pdf_reader.pages),
                    'file_size': os.path.getsize(pdf_path),
                    'file_name': os.path.basename(pdf_path)
                }
                
                # Try to get PDF metadata
                if pdf_reader.metadata:
                    metadata.update({
                        'title': pdf_reader.metadata.get('/Title', 'Unknown'),
                        'author': pdf_reader.metadata.get('/Author', 'Unknown'),
                        'creator': pdf_reader.metadata.get('/Creator', 'Unknown'),
                        'creation_date': pdf_reader.metadata.get('/CreationDate', 'Unknown')
                    })
                
                return metadata
                
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            return {
                'page_count': 0,
                'file_size': 0,
                'file_name': os.path.basename(pdf_path) if pdf_path else 'Unknown'
            }
    
    def batch_process_pdfs(self, pdf_directory: str) -> List[Dict[str, any]]:
        """
        Process multiple PDF files in a directory
        
        Args:
            pdf_directory (str): Directory containing PDF files
            
        Returns:
            List[Dict]: List of extracted data from all PDFs
        """
        try:
            if not os.path.exists(pdf_directory):
                raise FileNotFoundError(f"Directory not found: {pdf_directory}")
            
            pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                logger.warning(f"No PDF files found in directory: {pdf_directory}")
                return []
            
            results = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(pdf_directory, pdf_file)
                try:
                    extracted_data = self.extract_text_from_pdf(pdf_path)
                    results.append(extracted_data)
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file}: {str(e)}")
                    continue
            
            logger.info(f"Successfully processed {len(results)} out of {len(pdf_files)} PDF files")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = PDFProcessor()
    
    # Test with a sample PDF (you can replace this with an actual PDF path)
    sample_pdf = "app-1.pdf"
    
    if os.path.exists(sample_pdf):
        try:
            # Extract text from PDF
            result = processor.extract_text_from_pdf(sample_pdf)
            
            print(f"Processed PDF: {result['file_name']}")
            print(f"Number of pages: {result['page_count']}")
            print(f"Extraction method: {result['extraction_method']}")
            print(f"Total text length: {len(result['full_text'])} characters")
            print("\nFirst 500 characters of extracted text:")
            print(result['full_text'][:500])
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
    else:
        print(f"Sample PDF file '{sample_pdf}' not found. Please provide a valid PDF file for testing.")