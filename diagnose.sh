#!/bin/bash

# Diagnose dataset issues
# Can check specific samples or all samples

echo "Running dataset diagnostics..."
echo "================================================"

# Check all samples
python diagnose_data.py \
    --h5 processed_data/normalized_full_dataset.hdf5 \
    --split processed_data/train_val_test_split.json \
    --output diagnostic_report.txt

# Check specific problem sample with plot
echo -e "\nChecking S006 specifically with coverage plot..."
python diagnose_data.py \
    --h5 processed_data/normalized_full_dataset.hdf5 \
    --split processed_data/train_val_test_split.json \
    --sample S006 \
    --plot \
    --output S006_diagnostic.txt

echo "================================================"
echo "Diagnostics complete!"
echo "Check diagnostic_report.txt and S006_diagnostic.txt for details"
