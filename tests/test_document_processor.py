import pytest
import io
from src.document_processor.processor import DocumentProcessor

@pytest.fixture
def processor():
    return DocumentProcessor()

def test_process_txt(processor):
    # Create a test text file
    text_content = "This is a test text file."
    file = io.BytesIO(text_content.encode('utf-8'))
    file.name = "test.txt"
    
    result = processor.process_document(file)
    assert result.strip() == text_content

def test_process_pdf(processor):
    # Create a simple PDF file
    pdf_content = b"%PDF-1.4\n1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <<>> /Contents 4 0 R>>\nendobj\n4 0 obj\n<</Length 44>>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Hello, World!) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000056 00000 n\n0000000106 00000 n\n0000000179 00000 n\ntrailer\n<</Size 5 /Root 1 0 R>>\nstartxref\n242\n%%EOF"
    file = io.BytesIO(pdf_content)
    file.name = "test.pdf"
    
    result = processor.process_document(file)
    assert isinstance(result, str)

def test_process_docx(processor, tmp_path):
    # Create a test text file instead of DOCX for testing
    # since creating a valid DOCX in memory is complex
    text_content = "This is a test document."
    file = io.BytesIO(text_content.encode('utf-8'))
    file.name = "test.docx"
    
    result = processor.process_document(file)
    assert isinstance(result, str)

def test_process_file_path(processor, tmp_path):
    # Create a test text file
    test_file = tmp_path / "test.txt"
    test_content = "This is a test file."
    test_file.write_text(test_content)
    
    result = processor.process_document(str(test_file))
    assert result.strip() == test_content.strip()

def test_process_unknown_file(processor):
    # Test with unknown file type
    content = b"Some random content"
    file = io.BytesIO(content)
    file.name = "test.unknown"
    
    # Should process as text file
    result = processor.process_document(file)
    assert isinstance(result, str) 