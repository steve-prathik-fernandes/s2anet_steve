#!/usr/bin/env python3
"""
S2ANet Training Script with Python 3.10+ Fixes
"""

import sys
import os
import collections.abc
import torch
import torch.nn as nn

# Fix all collections issues for Python 3.10+
collections.Sequence = collections.abc.Sequence
collections.Mapping = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable

# Add current directory to path
sys.path.insert(0, '/root/s2anet')

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

def main():
    print("🚀 Starting S2ANet Training with Your Dataset!")
    print("=" * 60)
    
    # Load config
    config_file = '/root/s2anet/configs/dota/s2anet_r50_fpn_1x_dota_custom.py'
    cfg = Config.fromfile(config_file)
    
    # Set work directory
    cfg.work_dir = '/root/s2anet/work_dirs/s2anet_r50_fpn_1x_dota_custom'
    
    # Create work directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    print(f"📊 Dataset Information:")
    train_dataset = build_dataset(cfg.data.train)
    test_dataset = build_dataset(cfg.data.test)
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Test samples: {len(test_dataset)}")
    print(f"   - Classes: {len(train_dataset.CLASSES)}")
    
    print(f"\\n🔧 Model Configuration:")
    print(f"   - Model: {cfg.model.type}")
    print(f"   - Backbone: {cfg.model.backbone.type}")
    print(f"   - Head: {cfg.model.bbox_head.type}")
    print(f"   - Epochs: {cfg.total_epochs}")
    print(f"   - Learning Rate: {cfg.optimizer.lr}")
    
    print(f"\\n🏋️  Starting Training...")
    print(f"   - Work Directory: {cfg.work_dir}")
    print(f"   - GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    try:
        # Start training without validation
        train_detector(
            model=build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg),
            dataset=train_dataset,
            cfg=cfg,
            validate=False  # Disable validation
        )
        print("\\n🎉 Training Completed Successfully!")
        
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
