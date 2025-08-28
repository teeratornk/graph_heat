#!/bin/bash

# Diagnose test evaluation issues
# Specifically for samples that return 0 pairs

echo "Diagnosing test evaluation issues..."
echo "================================================"

# Check S006 specifically
echo "Checking S006 test evaluation..."
python diagnose_test_issue.py \
    --h5 processed_data/normalized_full_dataset.hdf5 \
    --split processed_data/train_val_test_split.json \
    --sample S006 \
    --batch_nodes 500

# Also check S003 for comparison (the other test sample)
echo -e "\n\nChecking S003 test evaluation for comparison..."
python diagnose_test_issue.py \
    --h5 processed_data/normalized_full_dataset.hdf5 \
    --split processed_data/train_val_test_split.json \
    --sample S003 \
    --batch_nodes 500

echo "================================================"
echo "Diagnostics complete!"
