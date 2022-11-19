import argparse
import glob
import logging
import os
from logging import INFO, getLogger

import albumentations as A
import albumentations.augmentations.geometric.functional
import numpy as np
from PIL import Image

AUGMENTATION_DIR = "aug_dataset"
logger = getLogger(__name__)
logging.basicConfig(level=INFO)


def flip(dataset_path: str):
    logger.info("flip aug starts.")
    transform = A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)])

    file_paths = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)
    augmented_dir = os.path.join(AUGMENTATION_DIR, "flip")
    os.makedirs(augmented_dir, exist_ok=True)
    augment_and_save(file_paths, augmented_dir, transform)
    logger.info("flip aug is fiinished.")


def rotate(dataset_path: str):
    logger.info("rotate aug starts.")
    transform = A.Compose(
        [
            albumentations.augmentations.geometric.RandomRotate90(p=1.0),
        ]
    )

    file_paths = glob.glob(os.path.join(dataset_path, "**", "*.png"), recursive=True)
    augmented_dir = os.path.join(AUGMENTATION_DIR, "flip")
    os.makedirs(augmented_dir, exist_ok=True)
    augment_and_save(file_paths, augmented_dir, transform)
    logger.info("rotate aug is finished.")


def blur(dataset_path: str):
    logger.info("gaussian_blur aug starts.")
    transform = A.Compose([A.GaussianBlur(blur_limit=[15, 17], p=1.0)])
    file_paths = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)
    augmented_dir = os.path.join(AUGMENTATION_DIR, "gaussian_blur")
    os.makedirs(augmented_dir, exist_ok=True)
    augment_and_save(file_paths, augmented_dir, transform)
    logger.info("gaussian_blur is finished.")


def downscale(dataset_path: str):
    logger.info("downscale aug starts.")
    transform = A.Compose([A.Downscale(scale_min=0.2, scale_max=0.3, p=1)])
    file_paths = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)
    augmented_dir = os.path.join(AUGMENTATION_DIR, "downscale")
    os.makedirs(augmented_dir, exist_ok=True)
    augment_and_save(file_paths, augmented_dir, transform)
    logger.info("downscale is finished.")


def brightness_contrast(dataset_path: str):
    logger.info("brightness_contrast aug starts.")
    transform = A.Compose([A.RandomBrightnessContrast([-0.4, 0.5], [-0.4, 0.5], p=1)])
    file_paths = glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True)
    augmented_dir = os.path.join(AUGMENTATION_DIR, "brightness_contrast")
    os.makedirs(augmented_dir, exist_ok=True)
    augment_and_save(file_paths, augmented_dir, transform)
    logger.info("brightness_contrast is finished.")


def augment_and_save(file_paths: list, augmented_dir: str, transform: A.Compose):
    for file_path in file_paths:
        basename_without_ext = os.path.basename(file_path)
        augmented_filepath = os.path.join(
            augmented_dir, os.path.dirname(file_path), basename_without_ext
        )
        os.makedirs(os.path.dirname(augmented_filepath), exist_ok=True)
        img = Image.open(file_path)
        numpy_image = np.array(img)
        transformed = transform(image=numpy_image)
        transformed = Image.fromarray(transformed["image"])
        transformed.save(augmented_filepath)


def split_aug_data():
    pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="augmentationを行うためのツール")
    p.add_argument("dataset_dir")
    args = p.parse_args()
    dataset_dir = args.dataset_dir
    rotate(dataset_dir)
