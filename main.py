
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
images_dir = dataset_root / "images"
root = dataset_root
train_dir = root / "train"
test_dir = root / "test"
valid_dir = root / 'validation'

random.seed(42)
if train_dir.exists() and any(train_dir.iterdir()):
    print("Train/test folders already exist. Skipping dataset split.")
else:
    print("Train/test folders not found. Creating dataset split...")
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)
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

        n = len(all_images)
        n_train = int(n * train_ratio)
        n_test = int(n * test_ratio)
        n_valid = n - n_train - n_test  # whatever is left

        train_images = all_images[:n_train]
        test_images = all_images[n_train:n_train+n_test]
        valid_images = all_images[n_train+n_test:]


        # Create class subfolders in train/ and test/
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)
        (valid_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Copy images into train/<class_name>/ and test/<class_name>/
        for img_path in train_images:
            dest = train_dir / class_name / img_path.name
            shutil.copy2(img_path, dest)

        for img_path in test_images:
            dest = test_dir / class_name / img_path.name
            shutil.copy2(img_path, dest)

        for img_path in valid_images:
            dest =  valid_dir / class_name / img_path.name
            shutil.copy2(img_path, dest)

    print("Done splitting into train/ and test/.")

some_class = os.listdir(valid_dir)[0]
print("Example class:", some_class)
print("Train files in that class:", os.listdir(train_dir / some_class))
print(len(os.listdir(train_dir / some_class)))




class TrashDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=transform)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]
    @property
    def classes(self):
        return self.dataset.classes

dataset = TrashDataset(root_dir=
                             train_dir,
                             )
print(len(dataset))

target_to_class = {v:k for k, v in ImageFolder(train_dir).class_to_idx.items()}
print(target_to_class)

transform = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
])

dataset = TrashDataset(root_dir=train_dir, transform=transform)

image, label = dataset[100]
print(image.shape)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for images, labels in dataloader:
  break
print(images.shape, labels.shape)

class SimpleTrashClassifier(nn.Module):
  def __init__(self, num_classes=30):
    super(SimpleTrashClassifier, self).__init__()
    # Where we define all the parts of the model
    self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
    self.features = nn.Sequential(*list(self.base_model.children())[:-1])
    enet_out_size = 1280
    # Make a classifier
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(enet_out_size, num_classes)
    )
  def forward(self, x):
    # Connect these parts and return the output
    x = self.features(x)
    output = self.classifier(x)
    return output

model = SimpleTrashClassifier(num_classes=30)


example_out = model(images)
example_out.shape

# lose function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(criterion(example_out, labels))

transform = transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
])

val_path = os.path.join(path,'valid')

train_dataset = TrashDataset(root_dir=train_dir, transform=transform)
val_dataset = TrashDataset(root_dir=valid_dir, transform=transform)
test_dataset = TrashDataset(root_dir=test_dir, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_epoch = 5
train_losses, val_losses = [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleTrashClassifier(num_classes=53)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model():
    for epoch in range(num_epoch):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_dataloader, desc='training loop'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # --- Validation phase ---
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_dataloader, desc='Validation loop'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(val_dataloader.dataset)
        val_losses.append(val_loss)

        # Print epoch stats
        print(f"Epoch {epoch+1}/{num_epoch} - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # ✅ Save once after all epochs
    torch.save(model.state_dict(), "trash_model.pth")
    print("Model saved to trash_model.pth")


# Testing


# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

torch.save(model.state_dict(), "trash_model.pth")

# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)
test_image = "websiteImagery/download (6).jpeg"
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

original_image, image_tensor = preprocess_image(test_image, transform)
probabilities = predict(model, image_tensor, device)

# Assuming dataset.classes gives the class names
class_names = dataset.classes
