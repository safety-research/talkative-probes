# proton options

import os
import threading
import torch

_launch_metadata_allow_sync = dict()  # Now per-device
_proton_lock = threading.Lock()


def _get_device_key():
    """Get current device for launch metadata key"""
    return torch.cuda.current_device() if torch.cuda.is_available() else 0


def launch_metadata_allow_sync():
    global _launch_metadata_allow_sync
    device_key = _get_device_key()
    
    if device_key not in _launch_metadata_allow_sync:
        with _proton_lock:
            if device_key not in _launch_metadata_allow_sync:
                _launch_metadata_allow_sync[device_key] = not (os.getenv("PROTON_LAUNCH_METADATA_NOSYNC") == "1")
    
    return _launch_metadata_allow_sync[device_key]


def set_launch_metadata_allow_sync(allow_sync: bool):
    global _launch_metadata_allow_sync
    with _proton_lock:
        device_key = _get_device_key()
        _launch_metadata_allow_sync[device_key] = allow_sync
