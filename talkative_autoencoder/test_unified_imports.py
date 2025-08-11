#!/usr/bin/env python3
"""Test script to verify the unified model manager imports and basic functionality"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing imports...")
    
    # Test importing the unified model manager
    from website.backend.app.model_manager import UnifiedModelManager
    print("✓ UnifiedModelManager imported successfully")
    
    # Test importing the API
    from website.backend.app import api as unified_api
    print("✓ Unified API imported successfully")
    
    # Test importing other required modules
    from website.backend.app.config import load_settings
    print("✓ Config imported successfully")
    
    # Test basic instantiation
    print("\nTesting basic instantiation...")
    settings = load_settings()
    print(f"✓ Settings loaded: {settings.devices}")
    
    # Create a unified model manager
    manager = UnifiedModelManager(settings)
    print(f"✓ UnifiedModelManager created with {manager.num_devices} devices")
    
    # Check system state
    state = manager.get_system_state()
    print(f"✓ System state retrieved: {state['num_devices']} devices")
    
    print("\nAll tests passed! ✓")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)