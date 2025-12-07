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
# Download latest version


path = kagglehub.dataset_download("alistairking/recyclable-and-household-waste-classification")

print("Path to dataset files:", path)
app = FastAPI()

print(os.listdir(path))
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