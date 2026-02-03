#!/usr/bin/env bash
set -euo pipefail

# ----- choose GPUs -----
export CUDA_VISIBLE_DEVICES="1,4,5,6"

# ----- determinism / reproducibility -----
export TF_CUDNN_DETERMINISTIC="1"

# ----- Delete prior runs logs -----
rm -rf /scratch/merlinf/repos/PINNs-Training-Dynamics/pdes/burgers/figures/*
rm -rf /scratch/merlinf/repos/PINNs-Training-Dynamics/pdes/burgers/logs/*
rm -rf /scratch/merlinf/repos/PINNs-Training-Dynamics/ckpts/burgers/*
# rm -rf /scratch/merlinf/repos/PINNs-Training-Dynamics/pdes/allen_cahn/figures/*
# rm -rf /scratch/merlinf/repos/PINNs-Training-Dynamics/pdes/allen_cahn/logs/*
# rm -rf /scratch/merlinf/repos/PINNs-Training-Dynamics/ckpts/allen_cahn/*

# ----- run -----
python -m src.main --config ./src/configs/default.py
