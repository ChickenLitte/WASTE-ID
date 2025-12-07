# api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import kagglehub
import timm
import os 
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import torch
from PIL import Image
# 1) Create FastAPI app
app = FastAPI()

# Allow frontend (adjust origin in real usage)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2) Model definition (same as in train.py)
class TrashClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TrashClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output


# 3) Load model and class names from checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("waste_model.pth", map_location=device)
class_names = checkpoint["class_names"]
num_classes = len(class_names)

model = TrashClassifier(num_classes=num_classes)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()

# 4) Same preprocess as training
transform = transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.ToTensor(),
])


@app.get("/")
def root():
    return {"message": "Trash classification API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        x = transform(image).unsqueeze(0).to(device)  # [1, 3, 128, 128]

        # Inference
        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())

        result = {
            "class_id": pred_idx,
            "class_name": class_names[pred_idx],
            "confidence": confidence,
        }

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
