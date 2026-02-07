import cv2, os, torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

GT_DIR = "/nlsasfs/home/gpucbh/vyakti22/saprak297/rclone-v1.73.0-linux-amd64/AI_SUMMIT_ENHANCEMENT"
PRED_DIR = "/nlsasfs/home/gpucbh/vyakti22/saprak297/rclone-v1.73.0-linux-amd64/image_enhanced"

psnr_list, ssim_list = [], []

for name in os.listdir(GT_DIR):
    gt = cv2.imread(os.path.join(GT_DIR, name))
    pred = cv2.imread(os.path.join(PRED_DIR, name))

    if gt is None or pred is None:
        continue

    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB) / 255.0
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB) / 255.0

    psnr_list.append(peak_signal_noise_ratio(gt, pred, data_range=1.0))
    ssim_list.append(structural_similarity(gt, pred, channel_axis=2))

print("Average PSNR:", np.mean(psnr_list))
print("Average SSIM:", np.mean(ssim_list))