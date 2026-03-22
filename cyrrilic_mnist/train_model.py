import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torch import nn, optim
from pathlib import Path
from PIL import Image, ImageOps


if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device = mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device = cuda")
else:
    device = torch.device("cpu")
    print("Device = cpu")

class CyrrilicDataset(Dataset):
  def __init__(self, path, transform):
    self.image_paths = []
    self.labels = []
    
    label_idx = 0
    for cls_dir in sorted(path.glob("*")):
          for img_path in cls_dir.glob("*.png"):
              self.image_paths.append(img_path)
              self.labels.append(label_idx)
          label_idx += 1

    self.transforms = transform

  def __len__(self):
    return len(self.image_paths)
  

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    label = self.labels[idx]
    image = Image.open(img_path).convert('RGBA')
    background = Image.new('RGB', image.size, (255, 255, 255))
    background.paste(image, (0, 0), image)
    image = ImageOps.invert(background)
    # plt.imshow(image)
    # plt.show()
    return self.transforms(image), label


class CyrrilicCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels = 1,
                           out_channels = 32,
                           kernel_size = 3, padding=1)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu1 = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2, 2)

    self.conv2 = nn.Conv2d(in_channels = 32, 
                           out_channels = 64,
                           kernel_size = 3, padding = 1)
    self.bn2 = nn.BatchNorm2d(64)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.MaxPool2d(2, 2)

    self.conv3 = nn.Conv2d(in_channels = 64,
                           out_channels = 128,
                           kernel_size = 3, padding=1)
    self.bn3 = nn.BatchNorm2d(128)
    self.relu3 = nn.ReLU()
    self.pool3 = nn.MaxPool2d(2, 2)

    self.conv4 = nn.Conv2d(in_channels = 128,
                           out_channels = 128,
                           kernel_size = 3, padding=1)
    self.bn4 = nn.BatchNorm2d(128)
    self.relu4 = nn.ReLU()
    self.pool4 = nn.MaxPool2d(2, 2)



    self.flat = nn.Flatten()
    
    self.fc1 = nn.Linear(8 * 8 * 128, 128)
    self.relu5 = nn.ReLU()
    self.fc2 = nn.Linear(128, 34)


  def forward(self, x):
    x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
    x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
    x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
    x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
    x = self.flat(x)
    x = self.relu5(self.fc1(x))
    x = self.fc2(x)
    return x


train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomAffine(3, (0.03, 0.03), (0.9, 1.0)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transforms = transforms.Compose([
   transforms.Resize((128, 128)),
   transforms.Grayscale(1),
   transforms.ToTensor(),
   transforms.Normalize((0.5,), (0.5,))
])



path = Path("./tmp/Cyrillic/")
model = CyrrilicCNN().to(device)
train_dataset = CyrrilicDataset(path, train_transforms)
test_dataset = CyrrilicDataset(path, test_transforms)

indices = list(range(len(train_dataset)))
labels = train_dataset.labels


train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels)

train_data = Subset(train_dataset, train_idx)
val_data = Subset(train_dataset, )
test_data = Subset(test_dataset, test_idx)


train_loader = DataLoader(train_data, batch_size = 16, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 16, shuffle=False)

total_params = sum(p.numel() for p in model.parameters()) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_loss = []
train_acc = []
num_epochs = 100

save_path = Path("./tmp")
model_path = save_path / "model.pth"
model_path.parent.mkdir(parents=True, exist_ok=True)

model.train()

if not model_path.exists():
  best_loss = float('inf')
  patience = 5
  counter = 0

  for epoch in range(num_epochs):
     model.train()
     run_loss = 0.0
     correct = 0.0
     total = 0.0

     for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        _, predict = torch.max(output, 1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()

     scheduler.step()
     epoch_loss = run_loss / len(train_loader)
     epoch_acc = 100 * correct / total
     train_loss.append(epoch_loss)
     train_acc.append(epoch_acc)

     print(f"\nEpoch - {epoch}\n loss - {epoch_loss}\n acc - {epoch_acc}\n")

     if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
        torch.save(model.state_dict(), model_path)
     else:
        counter += 1
        if counter >= patience:
           break
else:
  model.load_state_dict(torch.load(model_path, map_location = device))

model.load_state_dict(torch.load(model_path, map_location = device))


plt.figure()
plt.plot(train_loss, label='Train Loss')
plt.plot(train_acc, label='Train Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('train.png')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test accuracy: {100 * correct / total}')
