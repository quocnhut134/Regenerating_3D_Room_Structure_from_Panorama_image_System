#!/bin/bash
source /d/miniconda3/etc/profile.d/conda.sh
conda activate cs105

cd C:/Users/PC/Documents/college/CS105/code/LGT-Net

python infer_all.py \
    --cfg src/config/mp3d.yaml \
    --img_glob "C:/Users/PC/Documents/college/CS105/code/data/img/*.png" \
    --json_dir "C:/Users/PC/Documents/college/CS105/code/output/mp3d/json/outputs/1_raw_lgt/" \
    --img_dir "C:/Users/PC/Documents/college/CS105/code/output/mp3d/image/outputs/1_raw_lgt/" \
    --txt_dir "C:/Users/PC/Documents/college/CS105/code/output/mp3d/txt/outputs/1_raw_lgt/" \
    --post_processing manhattan \
    --batch_size 2