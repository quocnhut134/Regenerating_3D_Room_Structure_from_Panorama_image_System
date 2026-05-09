""" 
@Date: 2026/04/23
@description: Explicit Batch Inference for Multiple Images
"""
import json
import os
import argparse
import math
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob

from tqdm import tqdm
from PIL import Image
from config.defaults import merge_from_file, get_config
from models.build import build_model
from postprocessing.post_process import post_process
from preprocessing.pano_lsd_align import panoEdgeDetection, rotatePanorama
from utils.boundary import corners2boundaries
from utils.conversion import depth2xyz
from utils.logger import get_logger
from utils.misc import tensor2np_d, tensor2np
from visualization.boundary import draw_boundaries
import gc

def parse_option():
    parser = argparse.ArgumentParser(description='Panorama Layout Inference All')
    parser.add_argument('--img_glob', type=str, required=True)
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--post_processing', type=str, default='manhattan', choices=['manhattan', 'atalanta', 'original'])
    
    parser.add_argument('--json_dir', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--txt_dir', type=str, required=True)
    
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for model inference')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    args.mode = 'test'
    return args

def visualize_2d(img, dt_xyz, processed_xyz, dt_ratio, show=False, save_path=None):
    dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=img.shape[1])
    vis_img = draw_boundaries(img, boundary_list=dt_boundaries, boundary_color=[0, 1, 0])

    if processed_xyz is not None:
        dt_boundaries = corners2boundaries(dt_ratio, corners_xyz=processed_xyz, step=None, visible=False, length=img.shape[1])
        vis_img = draw_boundaries(vis_img, boundary_list=dt_boundaries, boundary_color=[1, 0, 0])

    if save_path:
        result = Image.fromarray((vis_img * 255).astype(np.uint8))
        result.save(save_path)
    return vis_img

def preprocess(img_ori, q_error=0.7, refine_iter=3, vp_cache_path=None):
    if os.path.exists(vp_cache_path):
        with open(vp_cache_path) as f:
            vp = [[float(v) for v in line.rstrip().split(' ')] for line in f.readlines()]
            vp = np.array(vp)
    else:
        _, vp, _, _, _, _, _ = panoEdgeDetection(img_ori, qError=q_error, refineIter=refine_iter)
        
    i_img = rotatePanorama(img_ori, vp[2::-1])

    if vp_cache_path is not None and not os.path.exists(vp_cache_path):
        with open(vp_cache_path, 'w') as f:
            for i in range(3):
                f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))
    return i_img, vp

def inference():
    if len(img_paths) == 0:
        logger.error('No images found')
        return

    # Tính toán tổng số Batch
    num_batches = math.ceil(len(img_paths) / args.batch_size)
    bar = tqdm(range(num_batches), ncols=100)
    
    # Lặp qua từng Batch (thay vì từng ảnh)
    for batch_idx in bar:
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(img_paths))
        current_batch_paths = img_paths[start_idx:end_idx]
        
        bar.set_description(f"Processing Batch {batch_idx+1}/{num_batches}")
        
        batch_imgs, batch_rgbs, batch_names, batch_full_names = [], [], [], []
        
        # 1. TIỀN XỬ LÝ (CPU) - Lần lượt cho các ảnh trong Batch hiện tại
        for img_path in current_batch_paths:
            name = os.path.basename(img_path).split('.')[0]
            json_save_path = os.path.join(args.json_dir, f"{name}_raw.json")
            if os.path.exists(json_save_path):
                print(f"skipping file name: {name}")
                continue
            ext = os.path.basename(img_path).split('.')[-1]
            
            img = np.array(Image.open(img_path).resize((1024, 512), Image.BICUBIC))[..., :3]
            if args.post_processing == 'manhattan':
                vp_path = os.path.join(args.txt_dir, f"{name}_vp.txt")
                img, vp = preprocess(img, vp_cache_path=vp_path)

            img_norm = (img/255.0).astype(np.float32)
            img_input = img_norm.transpose(2, 0, 1) # [3, H, W]
            
            batch_imgs.append(img_input)
            batch_rgbs.append(img_norm)
            batch_names.append(name)
            batch_full_names.append(f"{name}.{ext}")

        # 2. SUY LUẬN & HẬU XỬ LÝ
        if len(batch_imgs) > 0:
            run_batch_inference(batch_imgs, batch_rgbs, batch_names, batch_full_names)

@torch.no_grad()
def run_batch_inference(img_inputs, img_rgbs, names, full_names):
    model.eval()
    
    # Gộp thành 1 Tensor để chạy trên GPU [Batch_size, 3, 512, 1024]
    input_tensor = torch.from_numpy(np.stack(img_inputs)).to(args.device)
    
    # GPU Forward Pass (Chạy hàng loạt)
    dt = model(input_tensor)
    dt_np = tensor2np_d(dt)
    
    B = len(names)
    
    # 3. HẬU XỬ LÝ (CPU) - Tách batch ra để xử lý và lưu kết quả
    for i in range(B):
        name = names[i]
        img_rgb = img_rgbs[i]
        
        dt_depth = dt_np['depth'][i]
        
        # Đảm bảo ratio là 1 giá trị vô hướng (scalar)
        dt_ratio = dt_np['ratio'][i]
        if isinstance(dt_ratio, np.ndarray) and dt_ratio.size > 0:
             dt_ratio = dt_ratio[0]
             
        dt_xyz = depth2xyz(np.abs(dt_depth))
        
        processed_xyz = None
        if args.post_processing != 'original':
            # Chạy post process (dula-net / manhattan)
            processed = post_process(dt_depth[None, ...], type_name=args.post_processing)
            processed_xyz = processed[0]
            output_xyz = processed_xyz
        else:
            output_xyz = dt_xyz

        # Lưu Ảnh
        img_save_path = os.path.join(args.img_dir, f"{name}_pred.png")
        visualize_2d(img_rgb, dt_xyz, processed_xyz, dt_ratio, show=False, save_path=img_save_path)

        # Lưu JSON custom
        W, H = 1024, 512
        floor_boundary_uv, ceil_boundary_uv = corners2boundaries(dt_ratio, corners_xyz=output_xyz, step=None, visible=False, length=W)
        
        floor_y = (floor_boundary_uv[:, 1] * H).round().astype(int).tolist()
        ceil_y = (ceil_boundary_uv[:, 1] * H).round().astype(int).tolist()
        
        json_data = {
            "image_name": full_names[i],
            "image_size": [H, W],
            "floor_boundary": floor_y,
            "ceiling_boundary": ceil_y
        }
        
        json_save_path = os.path.join(args.json_dir, f"{name}_raw.json")
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=4)

    del input_tensor, dt, dt_np, img_inputs, img_rgbs
    if args.device=='cuda':
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    logger = get_logger()
    args = parse_option()
    config = get_config(args)

    if 'cuda' in args.device and not torch.cuda.is_available():
        logger.info(f'The {args.device} is not available, will use cpu ...')
        config.defrost()
        args.device = "cpu"
        config.TRAIN.DEVICE = "cpu"
        config.freeze()

    model, _, _, _ = build_model(config, logger)
    
    os.makedirs(args.json_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(args.txt_dir, exist_ok=True)

    img_paths = sorted(glob.glob(args.img_glob))
    inference()