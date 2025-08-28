#!/bin/bash

# Train Graph-DeepONet with recommended configurations
# This script runs the training with settings optimized for CPU training

echo "Starting Graph-DeepONet training with recommended settings..."
echo "================================================"

python train.py \
    --h5 processed_data/normalized_full_dataset.hdf5 \
    --split processed_data/train_val_test_split.json \
    --scalers processed_data/scalers.json \
    --outdir outputs \
    --seed 42 \
    --deterministic \
    \
    --q_dim 128 \
    --trunk_hidden 128 \
    --trunk_depth 2 \
    --glob_hidden 64 \
    --graph_hidden 64 \
    --graph_layers 1 \
    --use_fourier_time \
    --time_bands 4 \
    --film_global \
    \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --grad_clip 1.0 \
    --warmup_steps 200 \
    --cosine \
    --print_every 1 \
    \
    --samples_per_step 1 \
    --t_subsample_frac 0.2 \
    --node_subsample_frac 0.2 \
    --target_pairs_cap 5000 \
    --val_node_frac 0.25 \
    \
    --num_workers 0

echo "================================================"
echo "Training script launched!"
