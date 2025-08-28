#!/bin/bash

# Regenerate visualizations from existing test results
# Useful when you want to update plots without re-running evaluation

echo "Regenerating visualizations only..."
echo "================================================"

# Check if run directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./test_viz_only.sh <run_directory>"
    echo "Example: ./test_viz_only.sh outputs/train_results_20250827_154533"
    exit 1
fi

RUN_DIR=$1

# Check if run directory exists
if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory not found: $RUN_DIR"
    exit 1
fi

python test.py \
    --run_dir "$RUN_DIR" \
    --h5 processed_data/normalized_full_dataset.hdf5 \
    --split processed_data/train_val_test_split.json \
    --scalers processed_data/scalers.json \
    --viz_only

echo "================================================"
echo "Visualization regeneration complete!"
