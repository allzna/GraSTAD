#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python main.py \
  --data_dir "../datasets/TSB-AD-U" \
  --file_list "./File_List/TSB-AD-U-Eva.csv" \
  --output_dir "./results/uni" \
  --patch_size 96 \
  --num_iters 100 \
  --batch_size 512 \
  --lr 1e-4 \
  --seed 2027 \
  --use_revin
