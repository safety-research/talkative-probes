import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None
import torch # For torch.device type hint

def get_system_metrics(device: torch.device) -> dict:
    """Get current system performance metrics."""
    metrics = {}
    
    metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1) # Short interval
    metrics['memory_percent'] = psutil.virtual_memory().percent
    
    if device.type == 'cuda' and GPUtil is not None:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # If specific device index is available and valid
                if device.index is not None and device.index < len(gpus):
                    gpu = gpus[device.index]
                elif len(gpus) > 0 : # Fallback to first GPU if no index or invalid
                    gpu = gpus[0]
                else: # No GPUs found by GPUtil
                    gpu = None

                if gpu:
                    metrics['gpu_utilization'] = gpu.load * 100
                    metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                    metrics['gpu_temperature'] = gpu.temperature
        except Exception: # Broad exception for GPUtil failures
            pass # Silently ignore if GPUtil fails
    
    return metrics

def format_time(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 0: seconds = 0 # Handle potential negative ETA
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
