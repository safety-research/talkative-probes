import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None
import torch # For torch.device type hint
import gc

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


def show_new_cuda_tensors_and_memory_summary(reset_current_tensors=False):
    """
    Prints a summary of CUDA memory usage and details of new CUDA tensors
    that have appeared since the last time this function was called.
    Also reports total memory of all CUDA tensors and new CUDA tensors based on their reported sizes.
    """
    
    if '_previous_cuda_tensor_ids' not in globals():    
        global _previous_cuda_tensor_ids
        _previous_cuda_tensor_ids = set()

    if reset_current_tensors:
        _previous_cuda_tensor_ids = set()

    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tensor check.")
        return

    # Get current GPU memory usage from PyTorch
    allocated_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    print(f"PyTorch Allocated: {allocated_mb:.2f} MB")
    print(f"PyTorch Cached (Reserved): {reserved_mb:.2f} MB")

    # Get more detailed memory stats from PyTorch
    print(torch.cuda.memory_summary())

    # Header for the detailed tensor analysis section
    print("\n--- CUDA Tensor Analysis (via gc) ---")
    
    current_cuda_tensor_ids = set()
    new_tensors_details_list = [] 
    
    total_new_tensor_mem_mb = 0.0
    total_all_tensor_mem_mb = 0.0
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                obj_id = id(obj)
                current_cuda_tensor_ids.add(obj_id)
                
                current_tensor_mem_mb = 0.0
                try:
                    # Calculate memory in MB based on tensor size and element size
                    current_tensor_mem_mb = obj.element_size() * obj.nelement() / 1024**2
                except Exception as e_mem_calc:
                    # Log error in calculating memory for this specific tensor but continue
                    print(f"Warning: Could not calculate memory for tensor ID {obj_id} (type: {type(obj)}, size: {obj.size()}, dtype: {obj.dtype}): {e_mem_calc}")
                
                total_all_tensor_mem_mb += current_tensor_mem_mb

                if obj_id not in _previous_cuda_tensor_ids:
                    total_new_tensor_mem_mb += current_tensor_mem_mb
                    new_tensors_details_list.append(
                        f"New: {type(obj)}, Size: {obj.size()}, Memory: {current_tensor_mem_mb:.4f} MB, ID: {obj_id}, dtype: {obj.dtype}, shape: {obj.shape}"
                    )
        except ReferenceError:  # Handle weakrefs or objects that might have been collected
            pass
        except Exception as e_inspect_obj:  # Catch other potential errors during object inspection
            print(f"Warning: Error inspecting an object during tensor scan: {e_inspect_obj}")
            pass
    
    print("\n--- New CUDA Tensors Since Last Call (via gc) ---")
    if new_tensors_details_list:
        for detail_str in new_tensors_details_list:
            print(detail_str)
    else:
        print("No new CUDA tensors found since last call.")
    
    print(f"\nTotal memory of ALL identified CUDA tensors (sum of their sizes via gc): {total_all_tensor_mem_mb:.4f} MB")
    print(f"Total memory of NEW identified CUDA tensors (sum of their sizes via gc): {total_new_tensor_mem_mb:.4f} MB")

    # Update the set of known tensor IDs for the next call
    _previous_cuda_tensor_ids = current_cuda_tensor_ids
    print("--------------------------------------")


# %%
def find_tensor_attribute_by_id(target_id: int):
    """
    Searches through all live Python objects to find which object an attribute
    (that is a CUDA tensor) is associated with, given the tensor's ID.
    """

    found_associations = []

    for obj in gc.get_objects():
        # Check direct attributes
        try:
            for attr_name, attr_value in vars(obj).items():
                if torch.is_tensor(attr_value) and attr_value.is_cuda and id(attr_value) == target_id:
                    found_associations.append(f"Tensor ID {target_id} is attribute '{attr_name}' of object: {type(obj)} (obj_id: {id(obj)})")
                # Check if attribute is a list containing the tensor
                elif isinstance(attr_value, list):
                    for i, item in enumerate(attr_value):
                        if torch.is_tensor(item) and item.is_cuda and id(item) == target_id:
                            found_associations.append(f"Tensor ID {target_id} is item {i} in attribute list '{attr_name}' of object: {type(obj)} (obj_id: {id(obj)})")
                # Check if attribute is a dict containing the tensor
                elif isinstance(attr_value, dict):
                    for key, value in attr_value.items():
                        if torch.is_tensor(value) and value.is_cuda and id(value) == target_id:
                            found_associations.append(f"Tensor ID {target_id} is value for key '{key}' in attribute dict '{attr_name}' of object: {type(obj)} (obj_id: {id(obj)})")
        except (AttributeError, ReferenceError, TypeError): # Some objects don't have __dict__ or might be weakrefs
            pass
        except Exception: # Catch other potential errors during inspection
            pass
        
        # Check if the object itself is the tensor (less likely for attributes, but good for completeness)
        try:
            if torch.is_tensor(obj) and obj.is_cuda and id(obj) == target_id:
                found_associations.append(f"Tensor ID {target_id} is the object itself: {type(obj)} (obj_id: {id(obj)}), object: {obj}")
        except Exception:
            pass


    if found_associations:
        print(f"\n--- Associations for Tensor ID {target_id} ---")
        for assoc in found_associations:
            print(assoc)
    else:
        print(f"No direct attribute association found for CUDA tensor with ID {target_id}.")
    print("-------------------------------------------")
