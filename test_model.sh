#!/bin/bash

# Test Graph-DeepONet model on test set
# This script evaluates the best checkpoint and generates visualizations

echo "Testing Graph-DeepONet model..."
echo "================================================"

# Check if run directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./test_model.sh <run_directory>"
    echo "Example: ./test_model.sh outputs/train_results_20250827_154533"
    exit 1
fi

RUN_DIR=$1

# Check if run directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory not found: $RUN_DIR"
    exit 1
fi

# Check if best.pt exists
if [ ! -f "$RUN_DIR/best.pt" ]; then
    echo "Error: best.pt not found in $RUN_DIR"
    exit 1
fi

echo "Run directory: $RUN_DIR"

python test.py \
    --run_dir "$RUN_DIR" \
    --h5 processed_data/normalized_full_dataset.hdf5 \
    --split processed_data/train_val_test_split.json \
    --scalers processed_data/scalers.json \
    --batch_nodes 500 \
    --seed 42

echo "================================================"
echo "Testing complete!"
