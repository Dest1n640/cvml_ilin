import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageDraw2, ImageFont
import matplotlib.pyplot as plt
import random
import string
from pathlib import Path


class ImageDataset(Dataset):
  def __init__(self, mode=1 ,n=1000, size=256):
    super().__init__()
    self.n = n
    self.size = size
    self.mode = mode
    self.transforms = transforms.Compose([
      transforms.ToTensor()
    ])

  def _get_random_text(self, length=3):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

  def __len__(self):
    return self.n
  
  def __getitem__(self, index):
    image = Image.new("L", (self.size, self.size), color=255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = "ABC"
    x = 30
    y = 30
    if self.mode == 1:
      text = "ABD"
      x = random.randint(0, self.size - 40)
      y = random.randint(0, self.size - 20)
    elif self.mode == 2:
      text = self._get_random_text(3)
    elif self.mode == 3:
      text = self._get_random_text(random.randint(2, 10))
    elif self.mode == 4:
      text = self._get_random_text(random.randint(2, 10))
      x = random.randint(0, self.size - 40)
      y = random.randint(0, self.size - 20)

    draw.text((x, y), text, fill=0, font=font)
    tensor = self.transforms(image)
    return tensor, tensor
  
class Encoder(nn.Module):
  def __init__(self, latent_size=512):
    super().__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1, 32, 3, stride=2, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.Conv2d(32, 64, 3, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.Conv2d(64, 128, 3, stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),

      nn.Conv2d(128, 256, 3, stride=2, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU()
    )

    self.bottleneck = nn.Linear(256 * 16 * 16, latent_size)

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.bottleneck(x)
    return x

class Decoder(nn.Module):
  def __init__(self, latent_size=512):
    super().__init__()
    self.bottleneck = nn.Linear(latent_size, 256 * 16 * 16)
    self.features = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(), 

      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),

      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),

      nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.bottleneck(x)
    x = x.view(x.size(0), 256, 16, 16)
    x = self.features(x)
    return x

def train_model(mode, device):
  print(f"Обучение модели {mode}")
  encoder = Encoder()
  decoder = Decoder()
  encoder_params = sum(p.numel() for p in encoder.parameters())
  decoder_params = sum(p.numel() for p in decoder.parameters())

  print(encoder_params)
  print(decoder_params)

  data = ImageDataset(mode)
  data_loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=2)

  encoder.to(device)
  decoder.to(device)

  criterion = nn.MSELoss()
  optimizer = optim.Adam(list(encoder.parameters())+
                        list(decoder.parameters()))
  encoder.train()
  decoder.train()
  epochs = 20

  for epoch in range(epochs):
    epoch_loss = 0.0
    for imgs, _ in data_loader:
      imgs = imgs.to(device)
      optimizer.zero_grad()
      latent = encoder(imgs)
      output = decoder(latent)
      loss = criterion(output, imgs)
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch - {epoch}\n{avg_loss=:.2f}")

  torch.save(encoder.state_dict(), Path(f"tmp/encoder{mode}.pth"))
  torch.save(decoder.state_dict(), Path(f"tmp/decoder{mode}.pth"))

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
  print("Device ", str(device))
  Path("tmp").mkdir(exist_ok=True)

  while True:
      print("\n1. Текст фикс, позиция случайная")
      print("2. Текст случайный (фикс. длина), позиция фикс")
      print("3. Текст случайный (разная длина), позиция фикс")
      print("4. Текст случайный, позиция случайная")
      print("5. Обучить ВСЕ модели (1-4)")
      
      user_input = input("Введите режим (1-5): ")
      if user_input not in ["1", "2", "3", "4", "5"]:
          print("Выбран неверный режим, попробуйте снова.")
          continue
      
      mode_choice = int(user_input)
      
      if mode_choice == 5:
          for m in [1, 2, 3, 4]:
              train_model(m, device)
      else:
          train_model(mode_choice, device)
      
      print("\nОбучение завершено.")
      break

 