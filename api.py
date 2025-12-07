# api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import io
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm

# ---------------------------
# 1) Create FastAPI app
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 2) Model definition
# ---------------------------
class TrashClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(TrashClassifier, self).__init__()
        self.base_model = timm.create_model("efficientnet_b0", pretrained=False)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        enet_out_size = 1280
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------
# 3) Class names (30) and model size (53)
# ---------------------------
class_names = [
    "aerosol_cans",                 # 0
    "aluminum_food_cans",           # 1
    "aluminum_soda_cans",           # 2
    "cardboard_boxes",              # 3
    "cardboard_packaging",          # 4
    "clothing",                     # 5
    "coffee_grounds",               # 6
    "disposable_plastic_cutlery",   # 7
    "eggshells",                    # 8
    "food_waste",                   # 9
    "glass_beverage_bottles",       # 10
    "glass_cosmetic_containers",    # 11
    "glass_food_jars",              # 12
    "magazines",                    # 13
    "newspaper",                    # 14
    "office_paper",                 # 15
    "paper_cups",                   # 16
    "plastic_cup_lids",             # 17
    "plastic_detergent_bottles",    # 18
    "plastic_food_containers",      # 19
    "plastic_shopping_bags",        # 20
    "plastic_soda_bottles",         # 21
    "plastic_straws",               # 22
    "plastic_trash_bags",           # 23
    "plastic_water_bottles",        # 24
    "shoes",                        # 25
    "steel_food_cans",              # 26
    "styrofoam_cups",               # 27
    "styrofoam_food_containers",    # 28
    "tea_bags",                     # 29
]

NUM_LABEL_CLASSES = len(class_names)   # 30
NUM_MODEL_CLASSES = 53                 # what your checkpoint was trained with


# ---------------------------
# 4) Load trained weights
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
checkpoint_path = BASE_DIR / "trash_model.pth"

print("Looking for checkpoint at:", checkpoint_path)

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Model file not found at: {checkpoint_path}")

# This is the raw state dict with 53-class head weights
state_dict = torch.load(checkpoint_path, map_location=device)

# Build model with 53 outputs to match checkpoint
model = TrashClassifier(num_classes=NUM_MODEL_CLASSES)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ---------------------------
# 5) Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# ---------------------------
# 6) Routes
# ---------------------------
@app.get("/")
def root():
    return {"message": "Trash classification API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        x = transform(image).unsqueeze(0).to(device)  # [1, 3, 128, 128]

        with torch.no_grad():
            outputs = model(x)  # [1, 53]

            # 🔑 Only use the first 30 logits (the ones that correspond to your 30 classes)
            outputs = outputs[:, :NUM_LABEL_CLASSES]  # [1, 30]

            probs = torch.softmax(outputs, dim=1)[0]  # [30]
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
