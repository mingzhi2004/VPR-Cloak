import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from os.path import join
import cv2  
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import shutil
torch.backends.cudnn.benchmark = True

import parser
import commons
import datasets_ws
import network
import util
import faiss
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("attack_logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)

args.features_dim = 14 * 768
if args.eval_dataset_name.startswith("pitts"):
    args.infer_batch_size = args.infer_batch_size // 2

logging.info(f"Arguments: {args}")
logging.info(f"Output will be saved to {args.save_dir}")

model = network.CricaVPRNet()
model = model.to(args.device)

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)
else:
    logging.error("Please provide pretrained model path --resume")
    exit(1)

model = torch.nn.DataParallel(model)
model.eval()

eval_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, args.eval_dataset_name, "test")
logging.info(f"Test set: {eval_ds}")
logging.info(f"Dataset contains {len(eval_ds)} images in total")
logging.info(f"Database images: {eval_ds.database_num}")
logging.info(f"Query images: {eval_ds.queries_num}")

eval_ds.apply_transform = False
positives_per_query = eval_ds.get_positives()

test_output_dir = join(args.save_dir, "test")
queries_output_dir = join(test_output_dir, "queries")
database_output_dir = join(test_output_dir, "database")
os.makedirs(queries_output_dir, exist_ok=True)
os.makedirs(database_output_dir, exist_ok=True)

logging.info("Extracting database features and saving database images")
eval_ds.test_method = "hard_resize"
eval_ds.apply_transform = True

database_subset_ds = torch.utils.data.Subset(eval_ds, list(range(eval_ds.database_num)))
database_dataloader = DataLoader(dataset=database_subset_ds, batch_size=args.infer_batch_size,
                                 num_workers=args.num_workers, pin_memory=(args.device == "cuda"))

all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

with torch.no_grad():
    for inputs, indices in tqdm(database_dataloader, ncols=100):
        features = model(inputs.to(args.device))
        features = features.cpu().numpy()
        all_features[indices.numpy(), :] = features

        for idx in indices.numpy():
            img_path = eval_ds.images_paths[idx]
            img_filename = os.path.basename(img_path)
            shutil.copyfile(img_path, os.path.join(database_output_dir, img_filename))

database_features = all_features[:eval_ds.database_num]

faiss_index = faiss.IndexFlatL2(args.features_dim)
faiss_index.add(database_features)
del database_features

learning_rate = 8e-2
n_steps = 10
epsilon = 0.03
lambda_psnr = 0.1

def pil_collate(batch):
    images, indices = zip(*batch)
    return list(images), torch.tensor(indices)

def tensor_preprocess(imgs, eval_ds):
    if eval_ds.test_method == "hard_resize":
        imgs = torch.nn.functional.interpolate(imgs, size=eval_ds.resize, mode='bilinear', align_corners=False)
    elif eval_ds.test_method == "single_query":
        imgs = torch.nn.functional.interpolate(imgs, size=eval_ds.resize, mode='bilinear', align_corners=False)
    elif eval_ds.test_method == "central_crop":
        _, _, H, W = imgs.shape
        crop_h, crop_w = eval_ds.resize
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2
        imgs = imgs[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
    elif eval_ds.test_method == "five_crops":
        crop_size = min(eval_ds.resize)
        imgs = torch.nn.functional.interpolate(imgs, size=crop_size, mode='bilinear', align_corners=False)
        imgs = five_crops(imgs, crop_size)
    elif eval_ds.test_method == "nearest_crop" or eval_ds.test_method == "maj_voting":
        crop_size = min(eval_ds.resize)
        imgs = torch.nn.functional.interpolate(imgs, size=crop_size, mode='bilinear', align_corners=False)
        imgs = five_crops(imgs, crop_size)
    else:
        raise ValueError(f"Unknown test_method: {eval_ds.test_method}")

    mean = torch.tensor([0.485, 0.456, 0.406]).to(imgs.device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(imgs.device).view(1, -1, 1, 1)
    imgs = (imgs - mean) / std
    return imgs

def five_crops(imgs, crop_size):
    batch_size, C, H, W = imgs.shape
    tl = imgs[:, :, 0:crop_size, 0:crop_size]
    tr = imgs[:, :, 0:crop_size, W - crop_size:W]
    bl = imgs[:, :, H - crop_size:H, 0:crop_size]
    br = imgs[:, :, H - crop_size:H, W - crop_size:W]
    center = imgs[:, :, (H - crop_size) // 2:(H + crop_size) // 2, (W - crop_size) // 2:(W + crop_size) // 2]
    crops = torch.stack([tl, tr, bl, br, center], dim=1)
    crops = crops.view(-1, C, crop_size, crop_size)
    return crops

def calculate_psnr_loss(original, perturbed):
    mse = torch.mean((original * 255 - perturbed * 255) ** 2)
    if mse == 0:
        return torch.tensor(0.0).to(original.device)
    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
    return -psnr

def load_weight_maps(image_paths, target_size=(224, 224)):
    weight_maps = []
    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        weight_path = os.path.join("./Dinov2_weight/get_weighted_mslschallenge", f"{base_name}.npy")
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file {weight_path} does not exist")
            
        weight_map = np.load(weight_path)
        weight_map = (weight_map - np.min(weight_map)) / (np.max(weight_map) - np.min(weight_map) + 1e-8)
        weight_map = cv2.resize(weight_map, target_size, interpolation=cv2.INTER_LINEAR)
        weight_map = np.clip(weight_map, 0, 1)
        weight_maps.append(weight_map)
    
    return torch.tensor(np.array(weight_maps), dtype=torch.float32).unsqueeze(1).to(args.device)

logging.info("Starting attack")
eval_ds.apply_transform = False
queries_subset_ds = torch.utils.data.Subset(eval_ds, list(range(eval_ds.database_num, len(eval_ds))))
queries_dataloader = DataLoader(dataset=queries_subset_ds, batch_size=args.infer_batch_size,
                                num_workers=args.num_workers, shuffle=False, pin_memory=(args.device == "cuda"),
                                collate_fn=pil_collate)

resize_transform = transforms.Resize(args.resize)

for idx, (images_pil_batch, indices) in enumerate(tqdm(queries_dataloader, ncols=100)):
    batch_size = len(images_pil_batch)
    indices = indices.to(args.device)
    query_indices = indices - eval_ds.database_num

    processed_images = []
    flip_indices = []

    for i, img_pil in enumerate(images_pil_batch):
        width, height = img_pil.size
        if height > width:
            img_pil = img_pil.transpose(method=Image.TRANSPOSE)
            flip_indices.append(i)
        img_pil = resize_transform(img_pil)
        processed_images.append(img_pil)

    images = [transforms.ToTensor()(img_pil) for img_pil in processed_images]
    images = torch.stack(images).to(args.device)

    image_paths_batch = [eval_ds.images_paths[i.item()] for i in indices]
    weight_maps = load_weight_maps(image_paths_batch, target_size=images.shape[2:])
    
    delta = torch.zeros_like(images, requires_grad=True).to(args.device)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)

    positive_indices_batch = [positives_per_query[q_idx.item()] for q_idx in query_indices.cpu()]

    positive_features_list = []
    max_positives = 0
    for pos_indices in positive_indices_batch:
        pos_feats = all_features[pos_indices]
        positive_features_list.append(torch.tensor(pos_feats).to(args.device))
        if len(pos_indices) > max_positives:
            max_positives = len(pos_indices)

    for i in range(batch_size):
        pos_feats = positive_features_list[i]
        pos_feats_normalized = pos_feats / pos_feats.norm(p=2, dim=1, keepdim=True)
        positive_features_list[i] = pos_feats_normalized
        
        num_positives = pos_feats.size(0)
        if num_positives < max_positives:
            padding = torch.zeros((max_positives - num_positives, args.features_dim)).to(args.device)
            positive_features_list[i] = torch.cat([positive_features_list[i], padding], dim=0)

    positive_features_batch = torch.stack(positive_features_list)

    for step in range(n_steps):
        optimizer.zero_grad()
        
        with torch.no_grad():
            pixel_epsilon = weight_maps * epsilon  
            delta.data = torch.clamp(delta.data, -pixel_epsilon, pixel_epsilon)
            
        perturbed_images = images + delta
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        perturbed_images_processed = tensor_preprocess(perturbed_images, eval_ds)

        if eval_ds.test_method in ["five_crops", "nearest_crop", "maj_voting"]:
            batch_size_crops = perturbed_images_processed.size(0)
            features = model(perturbed_images_processed)
            if eval_ds.test_method == "five_crops":
                features = torch.stack(torch.split(features, 5)).mean(1)
            elif eval_ds.test_method == "nearest_crop" or eval_ds.test_method == "maj_voting":
                features = features.view(batch_size, 5, -1)
                features = features.mean(dim=1)
            else:
                raise ValueError(f"Unknown test_method: {eval_ds.test_method}")
        else:
            features = model(perturbed_images_processed)
        query_features = features / features.norm(p=2, dim=1, keepdim=True)

        query_features_expanded = query_features.unsqueeze(1)
        distances = torch.cdist(query_features_expanded, positive_features_batch)
        distances = distances.squeeze(1)

        mask = positive_features_batch.norm(dim=2) != 0
        valid_distances = distances[mask]
        
        loss_pos = valid_distances.mean()
        loss_psnr = calculate_psnr_loss(images, perturbed_images)
        loss_total = -loss_pos + lambda_psnr * loss_psnr

        loss_total.backward()
        optimizer.step()
        
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -pixel_epsilon, pixel_epsilon)
            
        with torch.no_grad():
            actual_psnr = -loss_psnr.item()
            
        logging.info(f"Batch {idx + 1}/{len(queries_dataloader)}, Step [{step + 1}/{n_steps}], "
                    f"L_total: {loss_total.item():.4f}, L_pos(distance): {loss_pos.item():.4f}, PSNR: {actual_psnr:.2f}dB")

    perturbed_images = images + delta
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    for i in range(batch_size):
        idx_item = indices[i].item()
        q_idx = idx_item - eval_ds.database_num
        img_path = eval_ds.images_paths[idx_item]
        img_filename = os.path.basename(img_path)
        perturbed_image = perturbed_images[i]
        perturbed_image_pil = transforms.ToPILImage()(perturbed_image.cpu())

        if i in flip_indices:
            perturbed_image_pil = perturbed_image_pil.transpose(method=Image.TRANSPOSE)

        perturbed_image_pil.save(os.path.join(queries_output_dir, img_filename))

logging.info(f"Attack completed, time elapsed: {str(datetime.now() - start_time)[:-7]}")
