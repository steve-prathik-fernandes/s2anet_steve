#!/usr/bin/env python3
"""
Basic test script to check if S2ANet can run without full CUDA extensions
"""

import sys
import os
import torch
import cv2
import numpy as np

print("Testing basic imports...")

try:
    print("✓ PyTorch version:", torch.__version__)
    print("✓ CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("✓ CUDA version:", torch.version.cuda)
        print("✓ GPU count:", torch.cuda.device_count())
        print("✓ Current GPU:", torch.cuda.current_device())
        print("✓ GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("✗ PyTorch/CUDA error:", e)

try:
    import mmcv
    print("✓ MMCV version:", mmcv.__version__)
except Exception as e:
    print("✗ MMCV error:", e)

try:
    import mmdet
    print("✓ MMDetection imported successfully")
except Exception as e:
    print("✗ MMDetection import error:", e)

# Test basic image loading
try:
    demo_image = "/root/s2anet/demo/demo.jpg"
    if os.path.exists(demo_image):
        img = cv2.imread(demo_image)
        print(f"✓ Demo image loaded: {img.shape}")
    else:
        print("✗ Demo image not found")
except Exception as e:
    print("✗ Image loading error:", e)

print("\nBasic environment test completed!")
