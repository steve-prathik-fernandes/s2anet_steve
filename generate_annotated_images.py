#!/usr/bin/env python3
"""
S2ANet Inference Script for Generating Annotated Images
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

from mmdet.apis import init_detector, inference_detector
from mmdet.core import rotated_box_to_poly_single

def generate_annotated_images(config_file, checkpoint_file, image_dir, output_dir):
    """Generate annotated images using trained model"""
    
    print("🔍 Generating Annotated Images with S2ANet...")
    print("=" * 60)
    
    # Initialize model
    print(f"📦 Loading model from: {checkpoint_file}")
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print(f"✅ Model loaded successfully!")
    print(f"   - Classes: {model.CLASSES}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"\\n📸 Processing {len(image_files)} images from: {image_dir}")
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        
        print(f"\\n🔍 Processing: {img_file}")
        
        # Run inference
        result = inference_detector(model, img_path)
        
        # Load image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(img_rgb)
        
        # Draw detections
        detections_count = 0
        if len(result) > 0 and len(result[0]) > 0:
            bboxes = result[0]
            scores = result[1]
            
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta']
            class_names = model.CLASSES
            
            for i, (bbox, score) in enumerate(zip(bboxes, scores)):
                if score > 0.3:  # Confidence threshold
                    detections_count += 1
                    # Convert to polygon
                    poly = rotated_box_to_poly_single(torch.tensor(bbox).unsqueeze(0))
                    poly = poly.numpy().reshape(-1, 2)
                    
                    # Draw polygon
                    color = colors[i % len(colors)]
                    polygon = Polygon(poly, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(polygon)
                    
                    # Add label with confidence
                    class_name = class_names[0]  # Assuming single class for now
                    ax.text(poly[0, 0], poly[0, 1], f'{class_name}: {score:.2f}', 
                           fontsize=10, color=color, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        ax.set_title(f'Inference Results: {img_file} (Detections: {detections_count})')
        ax.axis('off')
        
        # Save result
        output_path = os.path.join(output_dir, f'annotated_{img_file}')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Detections: {detections_count}")
        print(f"   ✅ Saved: {output_path}")
    
    print(f"\\n🎉 Annotated images saved to: {output_dir}")

def main():
    """Main function with default parameters"""
    config_file = 'configs/dota/s2anet_r50_fpn_1x_dota_custom.py'
    checkpoint_file = 'work_dirs/s2anet_r50_fpn_1x_dota_custom/latest.pth'
    image_dir = 'data/dota_1024/test_split/images/'
    output_dir = 'output/inference_annotations/'
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_file):
        print(f"❌ Checkpoint not found: {checkpoint_file}")
        print("\\n🏋️  Please train the model first:")
        print("   python tools/train.py configs/dota/s2anet_r50_fpn_1x_dota_custom.py")
        return
    
    generate_annotated_images(config_file, checkpoint_file, image_dir, output_dir)

if __name__ == "__main__":
    main()
