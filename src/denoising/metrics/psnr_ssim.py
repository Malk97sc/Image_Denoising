# src/metrics/psnr_ssim.py
import torch
import torch.nn.functional as F
import math
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))

def ssim_metric(img1, img2):
    return ssim(img1, img2)
