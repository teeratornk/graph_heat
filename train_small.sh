#!/bin/bash

# Train Graph-DeepONet with smaller configuration for quick testing
# This runs faster but with reduced capacity

echo "Starting Graph-DeepONet training (small config)..."
echo "================================================"

python train.py \
    --h5 processed_data/normalized_full_dataset.hdf5 \
    --split processed_data/train_val_test_split.json \
    --scalers processed_data/scalers.json \
    --outdir outputs \
    --seed 42 \
    --deterministic \
    \
    --q_dim 64 \
    --trunk_hidden 64 \
    --trunk_depth 2 \
    --glob_hidden 32 \
    --graph_hidden 32 \
    --graph_layers 1 \
    --use_fourier_time \
    --time_bands 2 \
    --film_global \
    \
    --epochs 20 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --grad_clip 1.0 \
    --warmup_steps 50 \
    --print_every 1 \
    \
    --samples_per_step 1 \
    --t_subsample_frac 0.1 \
    --node_subsample_frac 0.1 \
    --target_pairs_cap 1000 \
    --val_node_frac 0.1 \
    \
    --num_workers 0

echo "================================================"
echo "Training script launched (small config)!"
