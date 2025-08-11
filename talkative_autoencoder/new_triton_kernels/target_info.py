import torch
import triton
import threading

cached_capabilities = {}
_init_lock = threading.Lock()


def _get_device_key():
    """Get current device ID for cache key"""
    if torch.cuda.is_available():
        return f"cuda_{torch.cuda.current_device()}"
    return "cpu"


def _thread_safe_cache_init(cache_dict, cache_key, init_func):
    """Thread-safe cache initialization with double-check pattern"""
    if cache_key not in cache_dict:
        with _init_lock:
            if cache_key not in cache_dict:
                cache_dict[cache_key] = init_func()
    return cache_dict[cache_key]


def is_cuda():
    device_key = _get_device_key()
    cache_key = f"{device_key}_is_cuda"
    
    def _init_is_cuda():
        target = triton.runtime.driver.active.get_current_target()
        return False if target is None else target.backend == "cuda"
    
    return _thread_safe_cache_init(cached_capabilities, cache_key, _init_is_cuda)


def is_hip():
    if "is_hip" not in cached_capabilities:
        cached_capabilities["is_hip"] = torch.cuda.is_available() and bool(torch.version.hip)
    return cached_capabilities["is_hip"]


def is_hip_cdna3():
    if "is_hip_cdna3" not in cached_capabilities:
        target = triton.runtime.driver.active.get_current_target()
        cached_capabilities["is_hip_cdna3"] = (target is not None and target.backend == 'hip'
                                               and target.arch == 'gfx942')
    return cached_capabilities["is_hip_cdna3"]


def is_hip_cdna4():
    if "is_hip_cdna4" not in cached_capabilities:
        target = triton.runtime.driver.active.get_current_target()
        cached_capabilities["is_hip_cdna4"] = (target is not None and target.backend == 'hip'
                                               and target.arch == 'gfx950')
    return cached_capabilities["is_hip_cdna4"]


def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    if is_hip():
        return False
    
    device_key = _get_device_key()
    cache_key = f"{device_key}_cuda_capability"
    
    def _init_cuda_capability():
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability()
        else:
            return (0, 0)
    
    capability = _thread_safe_cache_init(cached_capabilities, cache_key, _init_cuda_capability)
    return capability >= (major, minor)


def get_cdna_version():
    """
    Gets the AMD architecture version, i.e. CDNA3 or CDNA4, currently
    only supports 3 (gfx942) or 4 (gfx950). Returns -1 if it is not AMD
    hardware or unsupported architecture
    """
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'hip':
        return -1
    if target.arch == 'gfx942':
        return 3
    if target.arch == 'gfx950':
        return 4
    return -1


def has_tma_gather():
    return cuda_capability_geq(10, 0)


def has_native_mxfp():
    return cuda_capability_geq(10, 0)


def num_sms(device=None):
    """Get SM count for specified device or current device"""
    if device is not None:
        if isinstance(device, torch.device):
            device_id = device.index if device.type == 'cuda' else 0
        elif isinstance(device, str):
            device_id = int(device.split(':')[1]) if ':' in device else 0
        else:
            device_id = device
    else:
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    return torch.cuda.get_device_properties(device_id).multi_processor_count
