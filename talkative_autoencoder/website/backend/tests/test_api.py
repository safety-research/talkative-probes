import pytest
import sys
import os
# Add the backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from datetime import datetime

@pytest.fixture
def mock_model_manager():
    """Mock ModelManager for testing"""
    manager = Mock()
    manager.queue = Mock()
    manager.queue.add_request = AsyncMock(return_value="test-request-id")
    manager.queue.get_status = Mock(return_value={
        "status": "completed",
        "result": {"data": [], "metadata": {}},
        "created_at": datetime.now()
    })
    manager.queue.queue = Mock()
    manager.queue.queue.qsize = Mock(return_value=0)
    manager.queue.active_requests = {}
    manager.analyzer = Mock()  # Model is loaded
    return manager

@pytest.fixture
def client(mock_model_manager):
    """Test client with mocked dependencies"""
    from app import main
    from app.websocket import manager as ws_manager
    
    # Store original value
    original_model_manager = main.model_manager
    
    # Set the mock
    main.model_manager = mock_model_manager
    
    # Create test client with lifespan disabled to avoid startup issues
    test_client = TestClient(main.app)
    
    yield test_client
    
    # Restore original value
    main.model_manager = original_model_manager

def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Consistency Lens API"

def test_analyze_endpoint_valid_request(client):
    """Test valid analysis request"""
    response = client.post("/analyze", json={
        "text": "Hello world",
        "options": {"batch_size": 16}
    })
    assert response.status_code == 200
    assert "request_id" in response.json()
    assert response.json()["status"] == "queued"

def test_analyze_endpoint_empty_text(client):
    """Test empty text validation"""
    response = client.post("/analyze", json={
        "text": "   "  # Just whitespace
    })
    assert response.status_code == 422

def test_analyze_endpoint_text_too_long(client):
    """Test text length validation"""
    response = client.post("/analyze", json={
        "text": "x" * 1001
    })
    assert response.status_code == 422

def test_health_check(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "gpu_available" in response.json()

def test_metrics_endpoint(client):
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "total_requests" in response.json()
    assert "queue_size" in response.json()

def test_status_endpoint(client):
    """Test status endpoint for existing request"""
    response = client.get("/status/test-request-id")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

def test_status_endpoint_not_found(client, mock_model_manager):
    """Test status endpoint for non-existent request"""
    mock_model_manager.queue.get_status.return_value = None
    response = client.get("/status/nonexistent")
    assert response.status_code == 404