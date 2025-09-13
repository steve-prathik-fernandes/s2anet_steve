#!/usr/bin/env python3
"""
S2ANet Training Script with Your Dataset
"""

import sys
import os
import collections.abc
import torch
import torch.nn as nn

# Fix collections.Sequence issue for Python 3.10+
collections.Sequence = collections.abc.Sequence

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
    print(f"   - Training samples: {len(build_dataset(cfg.data.train))}")
    print(f"   - Test samples: {len(build_dataset(cfg.data.test))}")
    print(f"   - Classes: {len(build_dataset(cfg.data.train).CLASSES)}")
    
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
        # Start training
        train_detector(
            model=build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg),
            dataset=build_dataset(cfg.data.train),
            cfg=cfg,
            validate=True
        )
        print("\\n🎉 Training Completed Successfully!")
        
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
