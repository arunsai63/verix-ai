import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import io
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestAPI:
    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @patch('app.main.orchestrator')
    def test_upload_documents(self, mock_orchestrator, client):
        mock_orchestrator.ingest_dataset = AsyncMock(return_value={
            "status": "success",
            "documents_processed": 1,
            "chunks_created": 5,
            "document_ids": ["doc1"]
        })
        
        files = [
            ("files", ("test.txt", io.BytesIO(b"test content"), "text/plain"))
        ]
        
        response = client.post(
            "/api/upload",
            files=files,
            data={"dataset_name": "test_dataset"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["dataset_name"] == "test_dataset"
    
    @patch('app.main.orchestrator')
    def test_query_documents(self, mock_orchestrator, client):
        mock_orchestrator.process_query = AsyncMock(return_value={
            "status": "success",
            "query": "test query",
            "answer": "test answer",
            "citations": [],
            "highlights": [],
            "confidence": "high",
            "role": "general",
            "sources_count": 5
        })
        
        response = client.post(
            "/api/query",
            json={
                "query": "test query",
                "role": "general",
                "max_results": 10
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["answer"] == "test answer"
    
    def test_query_empty_query(self, client):
        response = client.post(
            "/api/query",
            json={
                "query": "",
                "role": "general"
            }
        )
        
        assert response.status_code == 400
    
    @patch('app.main.Path')
    def test_list_datasets(self, mock_path, client):
        mock_dataset_dir = Mock()
        mock_dataset_dir.name = "test_dataset"
        mock_dataset_dir.is_dir.return_value = True
        mock_dataset_dir.glob.return_value = []
        mock_dataset_dir.stat.return_value.st_ctime = 1234567890
        
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.iterdir.return_value = [mock_dataset_dir]
        
        response = client.get("/api/datasets")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @patch('app.services.vector_store.VectorStoreService')
    def test_delete_dataset(self, mock_vector_store, client):
        mock_instance = Mock()
        mock_instance.delete_dataset.return_value = True
        mock_vector_store.return_value = mock_instance
        
        response = client.delete("/api/datasets/test_dataset")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    @patch('app.services.vector_store.VectorStoreService')
    def test_get_dataset_stats(self, mock_vector_store, client):
        mock_instance = Mock()
        mock_instance.get_dataset_stats.return_value = {
            "dataset_name": "test_dataset",
            "total_chunks": 10,
            "total_documents": 2
        }
        mock_vector_store.return_value = mock_instance
        
        response = client.get("/api/datasets/test_dataset/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["dataset_name"] == "test_dataset"
        assert data["total_chunks"] == 10