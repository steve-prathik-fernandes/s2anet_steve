#!/usr/bin/env python3
"""
S2ANet Ground Truth Annotation Visualization Script
"""

import sys
import os
import collections.abc
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Fix collections issues for Python 3.10+
collections.Sequence = collections.abc.Sequence
collections.Mapping = collections.abc.Mapping
collections.Iterable = collections.abc.Iterable

# Add current directory to path
sys.path.insert(0, '/root/s2anet')

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.core import rotated_box_to_poly_single

def visualize_ground_truth():
    """Visualize ground truth annotations from your dataset"""
    print("🔍 Visualizing Ground Truth Annotations...")
    print("=" * 50)
    
    # Load config and dataset
    config_file = '/root/s2anet/configs/dota/s2anet_r50_fpn_1x_dota_custom.py'
    cfg = Config.fromfile(config_file)
    train_dataset = build_dataset(cfg.data.train)
    
    # Create output directory
    os.makedirs('output/ground_truth_annotations', exist_ok=True)
    
    print(f"📊 Dataset Information:")
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Classes: {len(train_dataset.CLASSES)}")
    print(f"   - Class names: {train_dataset.CLASSES}")
    
    # Process each training sample
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        
        # Get image info
        img_meta = sample['img_meta'].data
        filename = os.path.basename(img_meta['filename'])
        img_shape = img_meta['img_shape']
        
        # Get annotations
        gt_bboxes = sample['gt_bboxes'].data.numpy()
        gt_labels = sample['gt_labels'].data.numpy()
        
        print(f"\\n📸 Processing {filename}:")
        print(f"   - Image shape: {img_shape}")
        print(f"   - Objects: {len(gt_bboxes)}")
        
        # Load original image
        img_path = img_meta['filename']
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_rgb)
        
        # Draw bounding boxes
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta']
        class_names = train_dataset.CLASSES
        
        for j, (bbox, label) in enumerate(zip(gt_bboxes, gt_labels)):
            # Convert rotated bbox to polygon
            # bbox format: [x, y, w, h, angle]
            try:
                # Ensure bbox is a tensor with correct shape
                if isinstance(bbox, torch.Tensor):
                    bbox_tensor = bbox.unsqueeze(0) if bbox.dim() == 1 else bbox
                else:
                    bbox_tensor = torch.tensor(bbox).unsqueeze(0)
                
                poly = rotated_box_to_poly_single(bbox_tensor)
                poly = poly.numpy().reshape(-1, 2)
            except Exception as e:
                print(f"   ⚠️  Error converting bbox {j}: {e}")
                continue
            
            # Create polygon patch
            color = colors[j % len(colors)]
            polygon = Polygon(poly, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(polygon)
            
            # Add label
            class_name = class_names[label - 1] if label > 0 else 'background'
            ax.text(poly[0, 0], poly[0, 1], f'{class_name}', 
                   fontsize=10, color=color, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        ax.set_title(f'Ground Truth Annotations: {filename}')
        ax.axis('off')
        
        # Save visualization
        output_path = f'output/ground_truth_annotations/{filename}'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {output_path}")
    
    print(f"\\n🎉 Ground truth annotations saved to: output/ground_truth_annotations/")

if __name__ == "__main__":
    visualize_ground_truth()
