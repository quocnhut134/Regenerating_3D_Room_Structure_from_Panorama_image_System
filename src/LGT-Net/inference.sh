#!/bin/bash
source /d/miniconda3/etc/profile.d/conda.sh
conda activate cs105

cd C:/Users/PC/Documents/college/CS105/code/LGT-Net

python inference.py \
    --cfg src/config/mp3d.yaml \
    --img_glob /c/Users/PC/Documents/college/CS105/code/data/img/AFimg0001.png \
    --output_dir C:/Users/PC/Documents/college/CS105/code/output \
    --post_processing manhattan
