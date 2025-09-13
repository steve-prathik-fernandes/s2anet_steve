#!/bin/bash

# Quick S2ANet Annotation Script
# Simple script to generate annotated images from your dataset

echo "🚀 Quick S2ANet Annotation Generator"
echo "===================================="

# Run the main annotation script
./generate_s2anet_annotations.sh

echo ""
echo "✅ Quick annotation generation completed!"
echo "📁 Check the output in: output/s2anet_annotations/"
echo ""
echo "Generated files:"
echo "  - Ground truth annotations with colored bounding boxes"
echo "  - Combined visualizations (GT + space for predictions)"
echo "  - Summary report"
echo ""
echo "To view the results:"
echo "  ls -la output/s2anet_annotations/"
echo "  open output/s2anet_annotations/ground_truth/  # View annotated images"
