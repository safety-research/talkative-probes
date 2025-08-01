# Request Cancellation Documentation

## How Request Cancellation Works

The system supports cancelling requests that are queued or in progress through WebSocket messages.

### Cancelling Queued Requests

To cancel a request that is still in the queue (status: "queued"):

1. Send a WebSocket message with type `cancel_request`:
```json
{
  "type": "cancel_request",
  "request_id": "your-request-id"
}
```

2. The system will:
   - Mark the request status as "cancelled"
   - Set an error message "Cancelled by user"
   - Update the completed timestamp
   - The request will be skipped when the processing loop encounters it

3. You'll receive a confirmation message:
```json
{
  "type": "request_cancelled",
  "request_id": "your-request-id"
}
```

### Interrupting Processing Requests

To interrupt a request that is currently being processed (status: "processing"):

1. Send a WebSocket message with type `interrupt`:
```json
{
  "type": "interrupt",
  "request_id": "your-request-id",
  "context": "analysis"  // or "generation"
}
```

2. The system will:
   - Mark the request status as "cancelled"
   - The progress callback will detect the cancellation and stop processing
   - Clean up resources associated with the request

3. You'll receive an error message:
```json
{
  "type": "error",
  "request_id": "your-request-id",
  "error": "Analysis interrupted by user",
  "context": "analysis"
}
```

### Limitations

- You cannot cancel requests that are already completed, failed, or cancelled
- Cancellation relies on periodic status checks during processing, so there may be a slight delay
- The actual request data remains in the asyncio.Queue until encountered by the processing loop

### Queue Status Updates

The system broadcasts queue status updates every 2 seconds via WebSocket:
```json
{
  "type": "queue_update",
  "queue_size": 5,
  "processing": 1,
  "queued_ids": ["req-123", "req-124", "req-125"]
}
```

Use this to monitor which requests are queued and their order.

### Memory Management

- Completed/failed/cancelled requests are automatically cleaned up after 1 hour
- The reset endpoint (`POST /reset`) clears all queued requests and reinitializes the queue
- Active requests dictionary is periodically cleaned to prevent memory leaks