import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path 
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

dataset_path = Path("./roads")

class RoadsDataset(Dataset):
  def __init__(self, images_list, masks_list, is_train=True):
    super().__init__()
    self.images = images_list
    self.masks = masks_list
    self.is_train = is_train

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, index):
    image = Image.open(self.images[index]).convert("RGB")
    image = np.array(image, dtype="f4") / 255.
    mask = Image.open(self.masks[index]).convert("L")
    mask = np.array(mask, dtype="f4")
    mask = (mask == 82).astype("f4")
    mask = np.expand_dims(mask, axis=0) #1, H, W

    if self.is_train and np.random.rand() > 0.5:
      image = np.flip(image, axis=1).copy()
      mask = np.flip(mask, axis=2).copy()

    image = torch.from_numpy(image.transpose(2, 0, 1)) # C, H, W
    mask = torch.from_numpy(mask)
    return image, mask

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, 1, 1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(),

      nn.Conv2d(out_channels, out_channels, 3, 1, 1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
    )
  def forward(self, x):
    return self.conv(x)
  
class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
    super().__init__()
    self.downscale = nn.ModuleList()
    self.upscale = nn.ModuleList()
    self.pool = nn.MaxPool2d(2, 2)
    for n in features:
      self.downscale.append(DoubleConv(in_channels, n))
      in_channels = n
    
    for n in reversed(features):
      self.upscale.append(nn.ConvTranspose2d(n * 2, n, 2, 2))
      self.upscale.append(DoubleConv(n * 2, n))

    self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
    self.result = nn.Conv2d(features[0], out_channels, 1)

  def forward(self, x):
    skips = []

    for ds in self.downscale:
      x = ds(x)
      skips.append(x)
      x = self.pool(x)

    x = self.bottleneck(x)

    skips = skips[::-1]
    for idx in range(0, len(self.upscale), 2):
      x = self.upscale[idx](x)
      skip = skips[idx // 2]
      cx = torch.cat((skip, x), dim=1)
      x = self.upscale[idx + 1](cx)
    return self.result(x)

class DiceLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, pred, target):
    pred_sig = torch.sigmoid(pred)
    p_area = pred_sig.view(-1)
    t_area = target.view(-1)
    intersection = (p_area * t_area).sum()
    return 1 - (2 * intersection + 1) / (pred_sig.sum() + t_area.sum() + 1)

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.mps.is_available()
                        else "cpu")

  dataset_path = Path('./roads')
  all_images = sorted(list((dataset_path / "images").glob("*.png")))
  all_masks = sorted(list((dataset_path / "masks").glob("*.png")))

  train_imgs, _, train_masks, _ = train_test_split(
    all_images, all_masks, test_size=0.2
  )

  train_dataset = RoadsDataset(train_imgs, train_masks, is_train=True)
  train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

  unet = UNet()
  unet.to(device)
  model_path = Path("tmp/unet.pth")
  model_path.parent.mkdir(parents=True, exist_ok=True)

  criterion = DiceLoss()

  optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, unet.parameters()),
    lr = 0.0001
  )

  epochs = 40
  best_loss = float('inf')
  curr_patience = 0
  patience = 3

  if not model_path.exists():
    for epoch in range(1, epochs):
      unet.train()
      for batch_idx, (images, labeles) in enumerate(train_dataloader):
        images = images.to(device)
        labeles = labeles.to(device)
        optimizer.zero_grad()
        output = unet(images)
        loss = criterion(output, labeles)
        loss.backward()
        optimizer.step()

      print(f"Epoch - {epoch}\nDiceloss - {loss}")  

      if loss < best_loss:
        curr_patience = 0
        best_loss = loss
        torch.save(unet.state_dict(), model_path)
      else:
        curr_patience += 1

      if curr_patience == patience:
        break
