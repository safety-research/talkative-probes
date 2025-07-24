import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from app.inference import InferenceQueue, ModelManager, ModelLoadError

@pytest.mark.asyncio
async def test_inference_queue_add_request():
    """Test adding request to queue"""
    queue = InferenceQueue(max_size=10)
    
    request_id = await queue.add_request("test text", {"batch_size": 32})
    
    assert request_id is not None
    assert request_id in queue.active_requests
    assert queue.active_requests[request_id]['text'] == "test text"
    assert queue.active_requests[request_id]['status'] == 'queued'

def test_inference_queue_get_status():
    """Test getting request status"""
    queue = InferenceQueue()
    
    # Manually add a request
    queue.active_requests['test-id'] = {
        'id': 'test-id',
        'status': 'processing',
        'text': 'test'
    }
    
    status = queue.get_status('test-id')
    assert status is not None
    assert status['status'] == 'processing'
    
    # Test non-existent request
    assert queue.get_status('nonexistent') is None

@pytest.mark.asyncio
async def test_model_manager_load_model_success():
    """Test successful model loading"""
    with patch('app.inference.os.path.exists', return_value=True):
        with patch('app.inference.torch.cuda.is_available', return_value=True):
            mock_analyzer = Mock()
            mock_analyzer.analyze_all_tokens = Mock(return_value=Mock())
            
            with patch('lens.analysis.analyzer_class.LensAnalyzer', return_value=mock_analyzer):
                manager = ModelManager("/fake/path.pt")
                await manager.load_model()
                
                assert manager.analyzer is not None

@pytest.mark.asyncio
async def test_model_manager_load_model_no_gpu():
    """Test model loading fails without GPU"""
    with patch('app.inference.os.path.exists', return_value=True):
        with patch('app.inference.torch.cuda.is_available', return_value=False):
            manager = ModelManager("/fake/path.pt", max_retries=1)
            
            with pytest.raises(ModelLoadError) as exc_info:
                await manager.load_model()
            
            assert "No GPU available" in str(exc_info.value)

@pytest.mark.asyncio
async def test_model_manager_load_model_missing_checkpoint():
    """Test model loading fails with missing checkpoint"""
    with patch('app.inference.os.path.exists', return_value=False):
        manager = ModelManager("/fake/path.pt", max_retries=1)
        
        with pytest.raises(ModelLoadError) as exc_info:
            await manager.load_model()
        
        assert "Checkpoint not found" in str(exc_info.value)