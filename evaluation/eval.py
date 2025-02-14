#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from monai.losses import DiceLoss
from PIL import Image  # used for FID conversion to RGB
from scipy.linalg import sqrtm
from skimage import io, transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------


def get_images_in_dir(directory, target_size, is_mask=False):
    """
    Load all PNG images from the given directory using skimage.io.imread.
    Each image is read as grayscale (one channel) and resized to (target_size, target_size).

    For regular images, bilinear interpolation is used and the result is converted to uint8.
    For segmentation masks, nearest-neighbor interpolation (order=0) is used.
    """
    print(f"Loading from {directory}...", end=" ", flush=True)
    images = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(".png"):
            img_path = os.path.join(directory, fname)
            try:
                with Image.open(img_path) as img:
                    # print(np.array(img).shape)
                    # Convert image to grayscale.
                    img = img.convert("L")
                    # Resize if necessary.
                    if img.size != (target_size, target_size):
                        resample = Image.NEAREST if is_mask else Image.BILINEAR
                        img = img.resize((target_size, target_size), resample=resample)
                    # Convert the PIL image to a NumPy array.
                    img = np.array(img)
                    images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        # break
    print(f"Loaded {len(images)} images.")
    return images


def calculate_psnr(img_list1, img_list2):
    """
    Calculate the average PSNR between two lists of images.
    Assumes images are NumPy arrays in the 0-255 range.
    """
    print("Calculating PSNR...")
    psnr_values = []
    for img1, img2 in zip(img_list1, img_list2):
        psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
        psnr_values.append(psnr)
    avg_psnr = np.mean(psnr_values)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    return avg_psnr


def calculate_ssim(img_list1, img_list2):
    """
    Calculate the average SSIM between two lists of images.
    """
    print("Calculating SSIM...")
    ssim_values = []
    for img1, img2 in zip(img_list1, img_list2):
        ssim, _ = structural_similarity(
            img1, img2, full=True, data_range=img1.max() - img1.min()
        )
        if not np.isnan(ssim):
            ssim_values.append(ssim)
    avg_ssim = np.mean(ssim_values)
    print(f"Average SSIM: {avg_ssim:.4f}")
    return avg_ssim


def calculate_fid(img_list1, img_list2):
    """
    Compute the Fréchet Inception Distance (FID) between two lists of images.
    For each image list:
      - Convert the grayscale images to 3-channel RGB.
      - Resize to 299x299 and apply normalization.
      - Extract features using a pretrained Inception v3 model.
    """
    print("Calculating FID...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.inception_v3(
        transform_input=False, weights=models.Inception_V3_Weights.IMAGENET1K_V1
    )
    # Replace final FC layer to output features.
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    # Define transform for FID.
    fid_transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def get_features(image_list, model, device, transform, batch_size=50):
        features = []
        for i in range(0, len(image_list), batch_size):
            batch_images = []
            for img in image_list[i : i + batch_size]:
                # If the image is grayscale, replicate channels to create an RGB image.
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                pil_img = Image.fromarray(img)
                pil_img = pil_img.convert("RGB")
                batch_images.append(transform(pil_img))
            batch_tensor = torch.stack(batch_images).to(device)
            with torch.no_grad():
                batch_features = model(batch_tensor)
            batch_features = batch_features.view(batch_features.size(0), -1)
            features.append(batch_features.cpu().numpy())
        features = np.concatenate(features, axis=0)
        return features

    features1 = get_features(img_list1, model, device, fid_transform)
    features2 = get_features(img_list2, model, device, fid_transform)

    mu1 = np.mean(features1, axis=0)
    mu2 = np.mean(features2, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)

    diff = mu1 - mu2
    diff_sq = np.sum(diff**2)

    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff_sq + np.trace(sigma1 + sigma2 - 2 * covmean)
    print(f"FID: {fid:.4f}")
    return fid


def calculate_dice(seg_list, seg_pred_list, num_classes):
    """
    Calculate the average Dice coefficient between two lists of segmentation masks
    using MONAI's DiceLoss.

    The segmentation masks are assumed to be single-channel NumPy arrays with integer class labels.
    This function converts both the ground truth and predicted masks to one-hot format,
    then computes the dice loss (which is 1 - dice coefficient), and finally converts it back.
    """
    print("Calculating Dice Score...")
    dice_scores = []
    # Create the DiceLoss function.
    dice_loss_fn = DiceLoss(include_background=True, to_onehot_y=False, softmax=False)
    for seg, seg_pred in zip(seg_list, seg_pred_list):
        # Convert NumPy arrays to torch tensors and add a batch dimension.
        seg_tensor = torch.from_numpy(seg).unsqueeze(0).long()  # Shape: [1, H, W]
        seg_pred_tensor = (
            torch.from_numpy(seg_pred).unsqueeze(0).long()
        )  # Shape: [1, H, W]

        # Convert to one-hot encoding. The output shape becomes [B, num_classes, H, W].
        seg_tensor_onehot = (
            F.one_hot(seg_tensor, num_classes=num_classes).permute(0, 3, 1, 2).float()
        )
        seg_pred_tensor_onehot = (
            F.one_hot(seg_pred_tensor, num_classes=num_classes)
            .permute(0, 3, 1, 2)
            .float()
        )

        # Calculate the dice loss. Since DiceLoss = 1 - dice coefficient,
        # we compute dice = 1 - loss.
        loss = dice_loss_fn(seg_pred_tensor_onehot, seg_tensor_onehot)
        dice = 1 - loss.item()
        dice_scores.append(dice)
    avg_dice = np.mean(dice_scores)
    print(f"Average Dice Score: {avg_dice:.4f}")
    return avg_dice


# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate image quality and segmentation performance."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to data. Expected folders: img, seg, img_gen, seg_pred, seg_pred_gen",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Target image size for both images and segmentation masks",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of segmentation classes (e.g., if classes are 0 to num_classes-1)",
    )
    args = parser.parse_args()

    # Construct expected folder paths.
    img_dir = os.path.join(args.dir, "img")
    img_gen_dir = os.path.join(args.dir, "img_gen")
    seg_dir = os.path.join(args.dir, "seg")
    seg_pred_dir = os.path.join(args.dir, "seg_pred")
    seg_pred_gen_dir = os.path.join(args.dir, "seg_pred_gen")

    # Load images and segmentation masks.
    img = get_images_in_dir(img_dir, target_size=args.img_size, is_mask=False)
    img_gen = get_images_in_dir(img_gen_dir, target_size=args.img_size, is_mask=False)
    seg = get_images_in_dir(seg_dir, target_size=args.img_size, is_mask=True)
    seg_pred = get_images_in_dir(seg_pred_dir, target_size=args.img_size, is_mask=True)
    seg_pred_gen = get_images_in_dir(
        seg_pred_gen_dir, target_size=args.img_size, is_mask=True
    )

    # Evaluate image quality metrics.
    avg_psnr = calculate_psnr(img, img_gen)
    avg_ssim = calculate_ssim(img, img_gen)
    fid_value = calculate_fid(img, img_gen)

    print("On gt: ", end=" ", flush=True)
    avg_dice = calculate_dice(seg, seg_pred, args.num_classes)
    print("On gen: ", end=" ", flush=True)
    avg_dice_gen = calculate_dice(seg, seg_pred_gen, args.num_classes)

    if not os.path.exists("eval_results.csv"):
        with open("eval_results.csv", "w") as f:
            table_header = [
                "Date",
                "Directory",
                "PSNR",
                "SSIM",
                "FID",
                "Dice",
                "Dice_gen",
            ]
            f.write(",".join(table_header) + "\n")
    # save as csv to file
    with open("eval_results.csv", "a") as f:
        date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        table_row = [
            date_time,
            args.dir,
            avg_psnr,
            avg_ssim,
            fid_value,
            avg_dice,
            avg_dice_gen,
        ]
        f.write(",".join(map(str, table_row)) + "\n")


if __name__ == "__main__":
    main()
