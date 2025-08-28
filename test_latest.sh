#!/bin/bash

# Test the most recent training run
# Automatically finds the latest train_results directory

echo "Finding latest training run..."
echo "================================================"

# Find the most recent train_results directory
LATEST_RUN=$(ls -td outputs/train_results_* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "Error: No training runs found in outputs/"
    exit 1
fi

echo "Latest run: $LATEST_RUN"

# Check if best.pt exists
if [ ! -f "$LATEST_RUN/best.pt" ]; then
    echo "Error: best.pt not found in $LATEST_RUN"
    exit 1
fi

python test.py \
    --run_dir "$LATEST_RUN" \
    --h5 processed_data/normalized_full_dataset.hdf5 \
    --split processed_data/train_val_test_split.json \
    --scalers processed_data/scalers.json \
    --batch_nodes 500 \
    --seed 42

echo "================================================"
echo "Testing complete!"
