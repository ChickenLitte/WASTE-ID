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

app = FastAPI()

@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    return JSONResponse(content={"filename": file.filename})