#!/usr/bin/env python3
"""
Test script to verify the dashboard can be imported without torch-related issues.
"""

import sys
import os

def test_import():
    """Test importing the dashboard module."""
    try:
        print("Testing dashboard import...")
        
        # Add the project root to Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        # Test importing key components
        print("✓ Importing basic modules...")
        import streamlit as st
        import mlflow
        import pandas as pd
        import numpy as np
        from PIL import Image
        
        print("✓ Basic imports successful")
        
        # Test torch import (optional)
        try:
            import torch
            from torchvision import transforms
            print("✓ PyTorch available")
            torch_available = True
        except ImportError:
            print("⚠ PyTorch not available (this is OK for basic functionality)")
            torch_available = False
        
        # Test the dashboard functions without running Streamlit
        print("✓ Testing dashboard functions...")
        
        # Mock streamlit functions for testing
        class MockSt:
            @staticmethod
            def error(msg): print(f"ERROR: {msg}")
            @staticmethod
            def warning(msg): print(f"WARNING: {msg}")
            @staticmethod
            def info(msg): print(f"INFO: {msg}")
        
        # Temporarily replace st with mock
        import src.mlflow.dashboard as dashboard
        original_st = dashboard.st
        dashboard.st = MockSt()
        
        try:
            # Test the transform function
            transform = dashboard.get_model_transform()
            if torch_available and transform is not None:
                print("✓ PyTorch transforms working")
            elif not torch_available:
                print("✓ Fallback mode ready")
            else:
                print("⚠ Transform creation failed")
            
            # Test fallback preprocessing
            from PIL import Image
            test_img = Image.new('RGB', (256, 256), color='red')
            fallback_result = dashboard.preprocess_image_fallback(test_img)
            expected_shape = (1, 3, 224, 224)
            if fallback_result.shape == expected_shape:
                print(f"✓ Fallback preprocessing working: {fallback_result.shape}")
            else:
                print(f"✗ Fallback preprocessing failed: got {fallback_result.shape}, expected {expected_shape}")
            
        finally:
            # Restore original st
            dashboard.st = original_st
        
        print("\n🎉 Dashboard import test completed successfully!")
        print("\nTo run the dashboard:")
        print("1. Using the runner script: python run_dashboard.py")
        print("2. Using streamlit directly: streamlit run src/mlflow/dashboard.py --server.fileWatcherType=none")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_import()
    sys.exit(0 if success else 1)
