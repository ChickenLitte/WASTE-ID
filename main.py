from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
# Download latest version
import kagglehub
import timm
import os 
from pathlib import Path
import random
import shutil
# Download latest version


path = Path(
    kagglehub.dataset_download("alistairking/recyclable-and-household-waste-classification")
    )

# print("Dataset root:", dataset_root)
# print("At root level:", os.listdir(dataset_root))   # ← should be ['images']

# images_dir = os.path.join(dataset_root, "images")
# print("Images dir:", images_dir)
# images_dir2 = os.path.join(images_dir,'images')
# print("Inside images:", os.listdir(images_dir2))  

dataset_root = Path(os.path.join(path,'images'))
train_ratio = 0.8

images_dir = dataset_root / "images"
root = dataset_root
train_dir = root / "train"
test_dir = root / "test"
train_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

random.seed(42)

for class_dir in images_dir.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name
    print(f"Processing class: {class_name}")

    # Get ALL image files under this class folder (including default/ and real_world/)
    all_images = [p for p in class_dir.rglob("*") if p.is_file()]
    print(f"{class_name}: found {len(all_images)} images")

    if not all_images:
        # Just in case something is wrong with this class, skip it
        continue

    random.shuffle(all_images)

    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]

    # Create class subfolders in train/ and test/
    (train_dir / class_name).mkdir(parents=True, exist_ok=True)
    (test_dir / class_name).mkdir(parents=True, exist_ok=True)

    # Copy images into train/<class_name>/ and test/<class_name>/
    for img_path in train_images:
        dest = train_dir / class_name / img_path.name
        shutil.copy2(img_path, dest)

    for img_path in test_images:
        dest = test_dir / class_name / img_path.name
        shutil.copy2(img_path, dest)

print("Done splitting into train/ and test/.")

some_class = os.listdir(train_dir)[0]
print("Example class:", some_class)
print("Train files in that class:", os.listdir(train_dir / some_class))
print(len(os.listdir(train_dir / some_class)))



# train_path = os.path.join(path, 'train')
# test_path = os.path.join(path, 'test')

# class PlayingCardDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.dataset = ImageFolder(root=root_dir, transform=transform)
#     def __len__(self):
#         return len(self.dataset)
#     def __getitem__(self, idx):
#         return self.dataset[idx]
#     @property
#     def classes(self):
#         return self.dataset.classes

# dataset = PlayingCardDataset(root_dir=
#                              train_path,
#                              )

# @app.post("/predict/")
# async def create_upload_file(file: UploadFile = File(...)):
#     return JSONResponse(content={"filename": file.filename})