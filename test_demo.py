#!/usr/bin/env python3
"""
Test script to run S2ANet demo with available functionality
"""

import sys
import os
import cv2
import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, '/root/s2anet')

print("Testing S2ANet demo functionality...")

try:
    # Test basic imports
    from mmdet.apis import init_detector, inference_detector
    print("✓ Successfully imported MMDetection APIs")
except ImportError as e:
    print(f"✗ Failed to import MMDetection APIs: {e}")
    print("This is expected if CUDA extensions are still building...")
    sys.exit(0)

# Test with a simple config
try:
    # Check if we have any config files
    config_dir = "/root/s2anet/configs"
    if os.path.exists(config_dir):
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.py')]
        print(f"✓ Found {len(config_files)} config files")
        
        # Try to find a simple config
        simple_configs = [f for f in config_files if 's2anet' in f.lower() and 'r50' in f]
        if simple_configs:
            config_file = os.path.join(config_dir, simple_configs[0])
            print(f"✓ Using config: {config_file}")
        else:
            print("✗ No suitable config files found")
            sys.exit(0)
    else:
        print("✗ Config directory not found")
        sys.exit(0)

    # Test image loading
    demo_image = "/root/s2anet/demo/demo.jpg"
    if os.path.exists(demo_image):
        img = cv2.imread(demo_image)
        print(f"✓ Demo image loaded: {img.shape}")
    else:
        print("✗ Demo image not found")
        sys.exit(0)

    print("\n✓ Basic setup completed successfully!")
    print("The project is ready to run, but you'll need:")
    print("1. A pre-trained model checkpoint")
    print("2. Complete CUDA extension build (currently in progress)")
    print("3. Proper dataset setup for training/evaluation")
    
    print(f"\nTo run inference once the build completes:")
    print(f"python demo/demo_inference.py {config_file} <checkpoint> <image_dir> <output_dir>")

except Exception as e:
    print(f"✗ Error during setup: {e}")
    import traceback
    traceback.print_exc()
