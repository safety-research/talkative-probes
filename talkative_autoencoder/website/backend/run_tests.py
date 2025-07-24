#!/usr/bin/env python
"""Simple test runner to avoid pytest directory scanning issues"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run tests
import subprocess
result = subprocess.run([
    sys.executable, "-m", "pytest", 
    "tests/test_api.py", 
    "-v", 
    "--tb=short",
    "--no-header",
    "-p", "no:cacheprovider"
], cwd=os.path.dirname(os.path.abspath(__file__)))

sys.exit(result.returncode)