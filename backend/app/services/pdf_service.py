from pypdf import PdfReader
from io import BytesIO

class PDFService:
    """
    Service responsible for handling PDF document operations.
    Extracts raw text from uploaded PDF files.
    """

    @staticmethod
    def extract_text(file_content: bytes) -> str:
        """
        Reads a PDF file from bytes and extracts all available text.
        
        Args:
            file_content (bytes): The binary content of the PDF file.
            
        Returns:
            str: The extracted text from all pages.
        """
        try:
            reader = PdfReader(BytesIO(file_content))
            extracted_text = ""
            
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
                    
            return extracted_text.strip()
        
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF document: {str(e)}")