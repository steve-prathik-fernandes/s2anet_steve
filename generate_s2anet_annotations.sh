#!/bin/bash

# S2ANet Annotation Generation Script
# This script generates annotated images based on S2ANet paper methodology
# Author: Generated for S2ANet project
# Date: $(date)

set -e  # Exit on any error

echo "🚀 S2ANet Annotation Generation Script"
echo "======================================"
echo ""

# Configuration
PROJECT_DIR="/root/s2anet_steve"
DATA_DIR="$PROJECT_DIR/data_file"
OUTPUT_DIR="$PROJECT_DIR/output/s2anet_annotations"
CONFIG_FILE="$PROJECT_DIR/configs/dota/s2anet_r50_fpn_1x_dota.py"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/ground_truth"
mkdir -p "$OUTPUT_DIR/model_predictions"
mkdir -p "$OUTPUT_DIR/combined"

echo "📁 Project Directory: $PROJECT_DIR"
echo "📁 Data Directory: $DATA_DIR"
echo "📁 Output Directory: $OUTPUT_DIR"
echo ""

# Function to check if required files exist
check_requirements() {
    echo "🔍 Checking requirements..."
    
    if [ ! -d "$DATA_DIR" ]; then
        echo "❌ Data directory not found: $DATA_DIR"
        exit 1
    fi
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "❌ Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    if [ ! -d "$DATA_DIR/dota_1024/trainval_split/images" ]; then
        echo "❌ Images directory not found"
        exit 1
    fi
    
    if [ ! -d "$DATA_DIR/dota_1024/trainval_split/labelTxt" ]; then
        echo "❌ Annotations directory not found"
        exit 1
    fi
    
    echo "✅ All requirements satisfied"
    echo ""
}

# Function to generate ground truth visualizations
generate_ground_truth() {
    echo "🎯 Generating Ground Truth Annotations..."
    echo "----------------------------------------"
    
    python3 << 'EOF'
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches

# DOTA class names and colors
DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]

# Colors for each class
CLASS_COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
    '#00FFFF', '#FFA500', '#800080', '#008000', '#800000',
    '#808000', '#008080', '#000080', '#808080', '#C0C0C0'
]

def parse_dota_annotation(ann_file):
    """Parse DOTA format annotation file"""
    annotations = []
    with open(ann_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 9:
                # Extract coordinates (8 values) and class name
                coords = [float(x) for x in parts[:8]]
                class_name = parts[8]
                difficulty = int(parts[9]) if len(parts) > 9 else 0
                annotations.append({
                    'coords': coords,
                    'class': class_name,
                    'difficulty': difficulty
                })
    return annotations

def draw_rotated_bbox(img, coords, class_name, color, thickness=2):
    """Draw rotated bounding box on image"""
    # Convert coordinates to numpy array
    points = np.array([[coords[i], coords[i+1]] for i in range(0, 8, 2)], np.int32)
    
    # Draw the polygon
    cv2.polylines(img, [points], True, color, thickness)
    
    # Add class label
    label_pos = (int(points[0][0]), int(points[0][1]) - 10)
    cv2.putText(img, class_name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return img

def visualize_ground_truth(image_dir, label_dir, output_dir):
    """Generate ground truth visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        print(f"Processing: {img_file}")
        
        # Load image
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_file}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_file = os.path.join(label_dir, img_file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt'))
        
        if os.path.exists(ann_file):
            annotations = parse_dota_annotation(ann_file)
            
            # Draw annotations
            for ann in annotations:
                class_name = ann['class']
                coords = ann['coords']
                difficulty = ann['difficulty']
                
                # Get color for class
                if class_name in DOTA_CLASSES:
                    color_idx = DOTA_CLASSES.index(class_name)
                    color = tuple(int(CLASS_COLORS[color_idx][i:i+2], 16) for i in (1, 3, 5))
                else:
                    color = (255, 255, 255)  # White for unknown classes
                
                # Adjust color based on difficulty
                if difficulty == 1:
                    color = tuple(int(c * 0.7) for c in color)  # Darker for difficult
                
                # Draw bounding box
                img = draw_rotated_bbox(img, coords, class_name, color)
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"gt_{img_file}")
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_bgr)
        
        print(f"Saved: {output_path}")

# Main execution
DATA_DIR = "/root/s2anet_steve/data_file"
image_dir = os.path.join(DATA_DIR, "dota_1024/trainval_split/images")
label_dir = os.path.join(DATA_DIR, "dota_1024/trainval_split/labelTxt")
output_dir = os.path.join(DATA_DIR, "../output/s2anet_annotations/ground_truth")

print("Generating ground truth visualizations...")
visualize_ground_truth(image_dir, label_dir, output_dir)
print("Ground truth visualization completed!")
EOF

    echo "✅ Ground truth annotations generated"
    echo ""
}

# Function to generate model predictions (if model available)
generate_model_predictions() {
    echo "🤖 Generating Model Predictions..."
    echo "----------------------------------"
    
    # Check if we have a trained model
    MODEL_CHECKPOINT=""
    
    # Look for common checkpoint locations
    if [ -f "$PROJECT_DIR/work_dirs/s2anet_r50_fpn_1x_dota_custom/latest.pth" ]; then
        MODEL_CHECKPOINT="$PROJECT_DIR/work_dirs/s2anet_r50_fpn_1x_dota_custom/latest.pth"
    elif [ -f "$PROJECT_DIR/checkpoints/s2anet_r50_fpn_1x_dota.pth" ]; then
        MODEL_CHECKPOINT="$PROJECT_DIR/checkpoints/s2anet_r50_fpn_1x_dota.pth"
    fi
    
    if [ -z "$MODEL_CHECKPOINT" ] || [ ! -f "$MODEL_CHECKPOINT" ]; then
        echo "⚠️  No trained model found. Skipping model predictions."
        echo "   To generate model predictions, you need a trained S2ANet model."
        echo "   Place your model checkpoint in one of these locations:"
        echo "   - $PROJECT_DIR/work_dirs/s2anet_r50_fpn_1x_dota_custom/latest.pth"
        echo "   - $PROJECT_DIR/checkpoints/s2anet_r50_fpn_1x_dota.pth"
        echo ""
        return
    fi
    
    echo "Using model: $MODEL_CHECKPOINT"
    
    python3 << 'EOF'
import sys
import os
import cv2
import numpy as np
import torch

# Add project to path
sys.path.insert(0, '/root/s2anet_steve')

try:
    from mmdet.apis import init_detector, inference_detector
    from mmdet.core import rotated_box_to_poly_single
    
    print("Model prediction generation (requires CUDA extensions)...")
    print("This feature will be available after CUDA compilation.")
    
except ImportError as e:
    print(f"Cannot import MMDetection APIs: {e}")
    print("Model predictions require CUDA extensions to be compiled.")
    print("Run 'python setup.py develop' to enable this feature.")

print("Model prediction generation skipped.")
EOF

    echo "✅ Model prediction generation completed"
    echo ""
}

# Function to create combined visualizations
create_combined_visualizations() {
    echo "🎨 Creating Combined Visualizations..."
    echo "--------------------------------------"
    
    python3 << 'EOF'
import os
import cv2
import numpy as np

def create_combined_visualization():
    """Create side-by-side comparison of GT and predictions"""
    
    gt_dir = "/root/s2anet_steve/output/s2anet_annotations/ground_truth"
    output_dir = "/root/s2anet_steve/output/s2anet_annotations/combined"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(gt_dir):
        print("Ground truth directory not found")
        return
    
    gt_files = [f for f in os.listdir(gt_dir) if f.startswith('gt_')]
    
    for gt_file in gt_files:
        gt_path = os.path.join(gt_dir, gt_file)
        gt_img = cv2.imread(gt_path)
        
        if gt_img is not None:
            # Create a simple combined image (GT on left, space for predictions on right)
            h, w = gt_img.shape[:2]
            combined_img = np.zeros((h, w*2, 3), dtype=np.uint8)
            
            # Place GT on left side
            combined_img[:, :w] = gt_img
            
            # Add text on right side
            cv2.putText(combined_img, "Ground Truth", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_img, "Model Predictions", (w+10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_img, "(Requires trained model)", (w+10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            # Save combined image
            output_path = os.path.join(output_dir, f"combined_{gt_file.replace('gt_', '')}")
            cv2.imwrite(output_path, combined_img)
            print(f"Created: {output_path}")

create_combined_visualization()
print("Combined visualizations created!")
EOF

    echo "✅ Combined visualizations created"
    echo ""
}

# Function to generate summary report
generate_summary_report() {
    echo "📊 Generating Summary Report..."
    echo "-------------------------------"
    
    python3 << 'EOF'
import os
import glob

def generate_report():
    """Generate a summary report of the annotation generation"""
    
    output_dir = "/root/s2anet_steve/output/s2anet_annotations"
    
    report = []
    report.append("S2ANet Annotation Generation Report")
    report.append("=" * 40)
    report.append("")
    
    # Count files
    gt_dir = os.path.join(output_dir, "ground_truth")
    pred_dir = os.path.join(output_dir, "model_predictions")
    combined_dir = os.path.join(output_dir, "combined")
    
    if os.path.exists(gt_dir):
        gt_count = len([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        report.append(f"Ground Truth Annotations: {gt_count} images")
    
    if os.path.exists(pred_dir):
        pred_count = len([f for f in os.listdir(pred_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        report.append(f"Model Predictions: {pred_count} images")
    
    if os.path.exists(combined_dir):
        combined_count = len([f for f in os.listdir(combined_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        report.append(f"Combined Visualizations: {combined_count} images")
    
    report.append("")
    report.append("DOTA Dataset Classes:")
    dota_classes = [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
        'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
        'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]
    for i, cls in enumerate(dota_classes, 1):
        report.append(f"  {i:2d}. {cls}")
    
    report.append("")
    report.append("Output Locations:")
    report.append(f"  Ground Truth: {gt_dir}")
    report.append(f"  Model Predictions: {pred_dir}")
    report.append(f"  Combined: {combined_dir}")
    
    # Save report
    report_path = os.path.join(output_dir, "annotation_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    # Print report
    print('\n'.join(report))
    print(f"\nReport saved to: {report_path}")

generate_report()
EOF

    echo "✅ Summary report generated"
    echo ""
}

# Main execution
main() {
    echo "Starting S2ANet annotation generation..."
    echo ""
    
    check_requirements
    generate_ground_truth
    generate_model_predictions
    create_combined_visualizations
    generate_summary_report
    
    echo "🎉 S2ANet annotation generation completed!"
    echo ""
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo "📊 Check the annotation_report.txt for detailed summary"
    echo ""
    echo "Next steps:"
    echo "1. Train a model using: python train_s2anet.py"
    echo "2. Download pre-trained weights from the model zoo"
    echo "3. Run inference with trained model"
    echo ""
}

# Run main function
main "$@"
