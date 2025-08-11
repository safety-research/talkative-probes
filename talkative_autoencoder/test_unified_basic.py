#!/usr/bin/env python3
"""Basic test of unified model manager without full app startup"""

import sys
import os
import asyncio

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def main():
    try:
        print("Testing unified model manager...")
        
        # Test imports
        from website.backend.app.model_manager import UnifiedModelManager
        from website.backend.app.config import load_settings
        
        print("✓ Imports successful")
        
        # Load settings
        settings = load_settings()
        print(f"✓ Settings loaded")
        
        # Create manager
        manager = UnifiedModelManager(settings)
        print(f"✓ Manager created with {manager.num_devices} devices")
        
        # Check system state
        state = manager.get_system_state()
        print(f"✓ System state: {len(state['devices'])} devices configured")
        
        # List groups
        print(f"✓ Groups loaded: {len(manager.groups)}")
        for group_id, group in list(manager.groups.items())[:3]:
            print(f"  - {group_id}: {group.name} ({len(group.models)} models)")
        
        # Test model locations
        test_model = "gemma_2b_it_FULL_9"
        location = manager.get_model_location(test_model)
        print(f"✓ Model {test_model} location: {'loaded' if location else 'not loaded'}")
        
        print("\n✅ All basic tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())