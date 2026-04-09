import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import cv2
import time
from PIL import Image
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def choose_device():
  if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device = mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device = cuda")
  else:
    device = torch.device("cpu")
    print("Device = cpu")
  return device

def run(model, loader, criterion, optimizer=None, device=torch.device("cpu")):
  trained = False
  if optimizer is not None:
    model.train()
    trained = True
  else:
    model.eval()
  
  total_loss, correct, total = 0, 0, 0

  with torch.set_grad_enabled(optimizer is not None):
    for images, labels in loader:
      images = images.to(device)
      labels = labels.to(device)
      logits = model(images)
      loss = criterion(logits, labels)
      if trained:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      total_loss += loss.item() * images.size(0)
      preds = logits.argmax(dim=1)
      correct += (preds == labels).sum().item()
      total += images.size(0)
  return total_loss / total, correct / total


def build_model(model ,device = torch.device("cpu"), model_path = Path('./tmp/model.pth')):
  for param in model.features.parameters():
      param.requires_grad = False

  features = model.classifier[1].in_features
  model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(features, 3)
  )
  if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
  return model.to(device)     

def train(model, device = torch.device("cpu"), model_path = Path("./tmp/model.pth")):
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
  )
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

  epoch = 15
  best_acc = 0.0

  for epoch in range(1, epoch):
    train_loss, train_acc = run(model, train_loader, criterion, optimizer, device=device)
    val_loss, val_acc = run(model, val_loader, criterion, device=device)
    scheduler.step()

    print(f"Epoch = {epoch}\ntrain_loss = {train_loss}\ntrain_acc = {train_acc}")
    print(f"val_loss = {val_loss}\nval_acc = {val_acc}\n")

    if val_acc > best_acc:
      best_acc = val_acc
      torch.save(model.state_dict(), model_path)


data_path = Path("./dataset/")

train_transform = transforms.Compose([
  transforms.Resize((240, 240)),
  transforms.RandomCrop(224),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(15),
  transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
  transforms.Resize((240, 240)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
])

train_ds = ImageFolder(data_path / "train", transform=train_transform)
val_ds = ImageFolder(data_path / "val", transform=val_transform)

train_loader = DataLoader(train_ds, 32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, 32, shuffle=False, num_workers=4)

if __name__ == "__main__":
  device = choose_device()

  model_dir = Path("./tmp")

  EffNet0_path = model_dir / "EffNet0.pth"
  EffNet1_path = model_dir / "EffNet1.pth"
  EffNet2_path = model_dir / "EffNet2.pth"


  EffNet0_weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
  EffNet1_weights = torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1
  EffNet2_weights = torchvision.models.EfficientNet_B2_Weights.IMAGENET1K_V1

  EffNet0 = build_model(torchvision.models.efficientnet_b0(EffNet0_weights), device, EffNet0_path)
  EffNet1 = build_model(torchvision.models.efficientnet_b1(EffNet1_weights), device, EffNet1_path)
  EffNet2 = build_model(torchvision.models.efficientnet_b2(EffNet2_weights), device, EffNet2_path)

  train(EffNet0, device, EffNet0_path)
  train(EffNet1, device, EffNet1_path)
  train(EffNet2, device, EffNet2_path)

  # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  # criterion = nn.CrossEntropyLoss()
  # optimizer = torch.optim.Adam(
  #   filter(lambda p: p.requires_grad, model.parameters()),
  #   lr=0.0001
  # )
  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

  # model_path = Path('./tmp/best.pth')

  # epoch = 10
  # best_acc = 0.0

  # for epoch in range(1, epoch):
  #   train_loss, train_acc = run(model, train_loader, criterion, optimizer, device=device)
  #   val_loss, val_acc = run(model, val_loader, criterion, device=device)
  #   scheduler.step()

  #   print(f"Epoch = {epoch}\ntrain_loss = {train_loss}\ntrain_acc = {train_acc}")
  #   print(f"val_loss = {val_loss}\nval_acc = {val_acc}\n")

  #   if val_acc > best_acc:
  #     best_acc = val_acc
  #     torch.save(model.state_dict(), model_path)

