"""Queue management for inference requests"""

import asyncio
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class InferenceQueue:
    """Manages the queue of inference requests"""
    
    def __init__(self, websocket_manager=None):
        self.queue = asyncio.Queue()
        self.active_requests: Dict[str, Dict[str, Any]] = {}
        self.websocket_manager = websocket_manager
        
    async def add_request(self, text: str, options: Dict[str, Any], request_type: str = "analyze") -> str:
        """Add a new request to the queue"""
        request_id = str(uuid.uuid4())
        
        # Extract dependency marker from options, if present
        depends_on = None
        try:
            depends_on = options.pop("depends_on", None)
        except Exception:
            depends_on = None

        request = {
            "id": request_id,
            "text": text,
            "options": options,
            "type": request_type,
            "status": "queued",
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }
        if depends_on:
            request["depends_on"] = depends_on
        
        self.active_requests[request_id] = request
        await self.queue.put((request_id, text, options, request_type))
        
        # Broadcast queue update
        await self._broadcast_queue_update()
        
        logger.info(f"Added {request_type} request {request_id} to queue (size: {self.queue.qsize()})")
        return request_id
        
    async def add_chained_requests(
        self,
        first_text: str,
        first_options: Dict[str, Any],
        first_type: str,
        second_text: str,
        second_options: Dict[str, Any],
        second_type: str,
        depends: bool = True,
    ) -> Tuple[str, str]:
        """Add two requests back-to-back so the second follows the first.
        Optionally marks the second as depending on the first's completion.
        """
        first_id = str(uuid.uuid4())
        second_id = str(uuid.uuid4())

        first_request = {
            "id": first_id,
            "text": first_text,
            "options": first_options,
            "type": first_type,
            "status": "queued",
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }

        second_request = {
            "id": second_id,
            "text": second_text,
            "options": second_options,
            "type": second_type,
            "status": "queued",
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }
        if depends:
            second_request["depends_on"] = first_id

        self.active_requests[first_id] = first_request
        self.active_requests[second_id] = second_request

        # Enqueue in order; FIFO processing ensures adjacency
        await self.queue.put((first_id, first_text, first_options, first_type))
        await self.queue.put((second_id, second_text, second_options, second_type))

        # Broadcast queue update once
        await self._broadcast_queue_update()

        logger.info(
            f"Added chained requests {first_id} ({first_type}) -> {second_id} ({second_type}) to queue"
        )
        return first_id, second_id
        
    async def add_group_switch_request(self, target_group_id: str, model_id: str, websocket=None) -> str:
        """Add a group switch request with high priority"""
        request_id = str(uuid.uuid4())
        
        request = {
            "id": request_id,
            "type": "group_switch",
            "target_group_id": target_group_id,
            "model_id": model_id,
            "status": "queued",
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
        }
        
        self.active_requests[request_id] = request
        
        # Add to regular queue (no priority)
        options = {"websocket": websocket, "target_group_id": target_group_id, "model_id": model_id}
        await self.queue.put((request_id, target_group_id, options, "group_switch"))
        
        # Broadcast queue update
        await self._broadcast_queue_update()
        
        logger.info(f"Added group switch request {request_id} to queue (position: {self.queue.qsize()})")
        return request_id
        
    async def get_next_request(self) -> Tuple[Optional[str], Optional[str], Optional[Dict], Optional[str]]:
        """Get the next request from the queue"""
        try:
            # Just use the regular queue - no priority
            request_id, text, options, request_type = await asyncio.wait_for(
                self.queue.get(),
                timeout=1.0
            )
            return request_id, text, options, request_type
        except asyncio.TimeoutError:
            return None, None, None, None
            
    def get_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a request"""
        return self.active_requests.get(request_id)
        
    def get_position_in_queue(self, request_id: str) -> int:
        """Get the position of a request in the queue"""
        # Create a list of queued requests in order
        queued_requests = [
            req_id for req_id, req in self.active_requests.items()
            if req["status"] == "queued"
        ]
        
        try:
            return queued_requests.index(request_id) + 1
        except ValueError:
            return 0
            
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        statuses = {"queued": 0, "processing": 0, "completed": 0, "failed": 0, "cancelled": 0}
        
        for request in self.active_requests.values():
            status = request.get("status", "unknown")
            if status in statuses:
                statuses[status] += 1
                
        return {
            "queue_size": self.queue.qsize(),
            "queued_requests": statuses["queued"],
            "processing_requests": statuses["processing"],
            "completed_requests": statuses["completed"],
            "failed_requests": statuses["failed"],
            "cancelled_requests": statuses["cancelled"],
            "total_active": len(self.active_requests),
        }
        
    async def _broadcast_queue_update(self):
        """Broadcast queue status to all connected clients"""
        if self.websocket_manager:
            stats = self.get_queue_stats()
            await self.websocket_manager.broadcast({
                "type": "queue_update",
                "queue_size": stats["queue_size"],
                "queued_requests": stats["queued_requests"],
                "processing_requests": stats["processing_requests"],
                "total_active": stats["total_active"],
            })
            
    async def _send_websocket_update(self, websocket, message: Dict[str, Any]):
        """Send an update to a specific websocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send websocket update: {e}")
            
    def clear_completed_requests(self, older_than_seconds: int = 3600):
        """Clear completed requests older than specified seconds"""
        now = datetime.utcnow()
        to_remove = []
        
        for request_id, request in self.active_requests.items():
            if request["status"] in ["completed", "failed", "cancelled"]:
                completed_at = request.get("completed_at")
                if completed_at and (now - completed_at).total_seconds() > older_than_seconds:
                    to_remove.append(request_id)
                    
        for request_id in to_remove:
            del self.active_requests[request_id]
            
        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old requests")