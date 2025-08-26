import pytest
import tempfile
import os
from pathlib import Path
from app.services.document_processor import DocumentProcessor


@pytest.fixture
def processor():
    return DocumentProcessor()


@pytest.fixture
def sample_text_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a sample document for testing.\n")
        f.write("It contains multiple lines of text.\n")
        f.write("We will use it to test document processing.")
        return f.name


@pytest.fixture
def sample_html_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write("<html><body>")
        f.write("<h1>Test Document</h1>")
        f.write("<p>This is a test paragraph.</p>")
        f.write("</body></html>")
        return f.name


class TestDocumentProcessor:
    def test_process_text_file(self, processor, sample_text_file):
        result = processor.process_file(
            sample_text_file,
            "test_dataset",
            {"author": "test"}
        )
        
        assert result is not None
        assert "content" in result
        assert "metadata" in result
        assert "chunks" in result
        assert result["metadata"]["dataset_name"] == "test_dataset"
        assert result["metadata"]["author"] == "test"
        assert len(result["chunks"]) > 0
        
        os.unlink(sample_text_file)
    
    def test_process_html_file(self, processor, sample_html_file):
        result = processor.process_file(
            sample_html_file,
            "test_dataset"
        )
        
        assert result is not None
        assert "Test Document" in result["content"]
        assert "test paragraph" in result["content"]
        assert result["metadata"]["file_extension"] == ".html"
        
        os.unlink(sample_html_file)
    
    def test_unsupported_file_type(self, processor):
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            f.write(b"unsupported content")
            
        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.process_file(f.name, "test_dataset")
        
        os.unlink(f.name)
    
    def test_file_not_found(self, processor):
        with pytest.raises(FileNotFoundError):
            processor.process_file("/nonexistent/file.txt", "test_dataset")
    
    def test_chunk_document(self, processor):
        content = "Line 1\n" * 100
        metadata = {"test": "metadata"}
        
        chunks = processor._chunk_document(
            content,
            metadata,
            chunk_size=50,
            chunk_overlap=10
        )
        
        assert len(chunks) > 1
        assert all("content" in chunk for chunk in chunks)
        assert all("metadata" in chunk for chunk in chunks)
        assert all(chunk["metadata"]["test"] == "metadata" for chunk in chunks)
    
    def test_batch_processing(self, processor):
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.txt',
                delete=False
            ) as f:
                f.write(f"Document {i} content")
                files.append(f.name)
        
        results = processor.process_batch(files, "test_dataset")
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert f"Document {i}" in result["content"]
        
        for f in files:
            os.unlink(f)
    
    def test_extract_citations(self, processor):
        content = """
        This is a document with citations.
        Source: Smith et al., 2023
        Another line with references.
        Citation: Important paper
        Reference: Another important work
        """
        
        citations = processor.extract_citations(content)
        
        assert len(citations) > 0
        assert any("Smith et al., 2023" in c["text"] for c in citations)
        assert any("Important paper" in c["text"] for c in citations)