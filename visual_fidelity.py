import os
import argparse
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings

warnings.filterwarnings("ignore")

loss_fn = lpips.LPIPS(net='alex')
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()


def load_and_resize_image(image_path, size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.Resampling.LANCZOS)
    return np.array(image)


def calculate_lpips(image1, image2):
    image1_tensor = lpips.im2tensor(image1)
    image2_tensor = lpips.im2tensor(image2)
    
    if torch.cuda.is_available():
        image1_tensor = image1_tensor.cuda()
        image2_tensor = image2_tensor.cuda()
    
    return loss_fn(image1_tensor, image2_tensor).item()


def calculate_psnr(image1, image2):
    return psnr(image1, image2, data_range=255)


def calculate_ssim(image1, image2):
    return ssim(image1, image2, data_range=255, channel_axis=-1, win_size=7)


def compute_metrics(reference_dir, target_dir, image_size=(224, 224)):
    reference_images = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    target_images = sorted([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    if len(reference_images) != len(target_images):
        raise ValueError(f"Number of images mismatch: {len(reference_images)} vs {len(target_images)}")

    lpips_scores = []
    psnr_scores = []
    ssim_scores = []

    for ref_name, target_name in zip(reference_images, target_images):
        ref_path = os.path.join(reference_dir, ref_name)
        target_path = os.path.join(target_dir, target_name)

        ref_img = load_and_resize_image(ref_path, size=image_size)
        target_img = load_and_resize_image(target_path, size=image_size)

        if ref_img.shape != (*image_size, 3) or target_img.shape != (*image_size, 3):
            raise ValueError(f"Image size must be {image_size[0]}x{image_size[1]}: {ref_name}, {target_name}")

        lpips_scores.append(calculate_lpips(ref_img, target_img))
        psnr_scores.append(calculate_psnr(ref_img, target_img))
        ssim_scores.append(calculate_ssim(ref_img, target_img))

    avg_lpips = round(np.mean(lpips_scores), 3)
    avg_psnr = round(np.mean(psnr_scores), 3)
    avg_ssim = round(np.mean(ssim_scores), 3)

    print(f"Average LPIPS: {avg_lpips}")
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")
    
    return avg_lpips, avg_psnr, avg_ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate LPIPS, PSNR, and SSIM between two image directories")
    parser.add_argument("--reference_dir", type=str, required=True, help="Path to reference images directory")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to target images directory")
    parser.add_argument("--image_size", type=int, default=224, help="Image resize dimension (default: 224)")
    
    args = parser.parse_args()
    
    compute_metrics(args.reference_dir, args.target_dir, image_size=(args.image_size, args.image_size))