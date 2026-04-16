import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def choose_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device = cuda")
    elif torch.mps.is_available():
        device =torch.device("mps")
        print("Device = mps")
    else:
        device = torch.device("cpu")
        print("Device = cpu")
    return device

device = choose_device()
classes = ["square", "circle", "triangle"]

class ShapesDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = []
        for cls_name in classes:
            img_dir = root / cls_name / "images"
            meta_dir = root / cls_name / "labels"
            for img_path in sorted(img_dir.glob("*.png")):
                labels_path = meta_dir / (img_path.stem + ".txt")
                self.images.append((img_path, labels_path))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, labels_path = self.images[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            tensor = self.transform(Image.fromarray(img))
        else:
            tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        cls, x, y, w, h = map(float, labels_path.read_text().split())
        bbox = torch.tensor([x, y, w, h], dtype=torch.float32)

        return tensor, int(cls), bbox


class SimpleDetector(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = nn.Sequential(
           # nn.Conv2d(3, 6, 5),
            # nn.AvgPool2d(2, 2),

            # nn.Conv2d(6, 16, 5),
            # nn.AvgPool2d(2, 2),

            # nn.Conv2d(16, 120, 5),
            nn.Conv2d(3, 32, (3, 3),padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, (3, 3),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, (3, 3),padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, (3, 3),padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d(2),
        )
        self.cls_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.bbox_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.Dropout(0.3)
            # Для регрессии координат Dropout вреден, так как он вносит случайный шум
        )
        self.cls_head = nn.Linear(128, num_classes)
        self.bbox_head = nn.Linear(512, 4)

    def forward(self, x):
        features = self.backbone(x)
        features_cls = self.cls_fc(features)
        features_bbox = self.bbox_fc(features)
        return self.cls_head(features_cls), torch.sigmoid(self.bbox_head(features_bbox))


def giou_loss(pred, target):
    p_x1 = pred[:, 0] - pred[:, 2] / 2
    p_y1 = pred[:, 1] - pred[:, 3] / 2
    p_x2 = pred[:, 0] + pred[:, 2] / 2
    p_y2 = pred[:, 1] + pred[:, 3] / 2

    t_x1 = target[:, 0] - target[:, 2] / 2
    t_y1 = target[:, 1] - target[:, 3] / 2
    t_x2 = target[:, 0] + target[:, 2] / 2
    t_y2 = target[:, 1] + target[:, 3] / 2

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_t = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)
    union = area_p + area_t - inter

    iou = inter / (union + 1e-7)

    c_x1 = torch.min(p_x1, t_x1)
    c_y1 = torch.min(p_y1, t_y1)
    c_x2 = torch.max(p_x2, t_x2)
    c_y2 = torch.max(p_y2, t_y2)
    area_c = (c_x2 - c_x1).clamp(min=0) * (c_y2 - c_y1).clamp(min=0)

    giou = iou - (area_c - union) / (area_c + 1e-7)
    return (1 - giou).mean()


def detection_loss(cls_pred, bbox_pred, cls_targets, bbox_targets, lambda_bbox=100.0):
    loss_cls = F.cross_entropy(cls_pred, cls_targets)
    loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_targets)
    loss_bbox += giou_loss(bbox_pred, bbox_targets)
    return loss_cls + lambda_bbox * loss_bbox, loss_cls, loss_bbox


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

def make_dataset(root):
  train_ds = ShapesDataset(root / "train", transform=transform)
  val_ds = ShapesDataset(root / "val", transform=transform)

  train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=0) #Поставил True что бы в батче были разные фигуры
  return train_loader, val_loader

def train(train_loader, val_loader, save_dir):
  model = SimpleDetector(num_classes=len(classes)).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

  epochs = 100
  save_path = save_dir / "best.pt"
  accuracy_threshold = 0.95
  no_improve_count = 5

  history = defaultdict(list)
  best_acc = 0.0
  best_val_loss = float('inf')
  best_acc = 0.0
  no_improve = 0

  if save_path.exists():
      model.load_state_dict(torch.load(save_path))
  else:
      for epoch in range(1, epochs + 1):
          model.train()
          train_loss = train_cls = train_box = 0.0
          for images, cls_t, bbox_t in train_loader:
              images, cls_t, bbox_t = (
              images.to(device),
              cls_t.to(device),
              bbox_t.to(device),
          )
              optimizer.zero_grad()
              cls_pred, bbox_pred = model(images)
              loss, lc, lb = detection_loss(cls_pred, bbox_pred, cls_t, bbox_t)
              loss.backward()
              optimizer.step()
              train_loss += loss.item()
              train_cls += lc.item()
              train_box += lb.item()

          n = len(train_loader)
          history["train_loss"].append(train_loss / n)
          history["train_cls"].append(train_cls / n)
          history["train_box"].append(train_box / n)

          model.eval()
          val_loss = correct = total = 0.0
          with torch.no_grad():
              for images, cls_t, bbox_t in val_loader:
                  images, cls_t, bbox_t = (
                  images.to(device),
                  cls_t.to(device),
                  bbox_t.to(device),
              )
                  cls_pred, bbox_pred = model(images)
                  print(bbox_pred[0], bbox_t[0])
                  loss, _, _ = detection_loss(cls_pred, bbox_pred, cls_t, bbox_t)
                  val_loss += loss.item()
                  correct += (cls_pred.argmax(1) == cls_t).sum().item()
                  total += cls_t.size(0)

          val_acc = correct / total
          history["val_loss"].append(val_loss / len(val_loader))
          history["val_acc"].append(val_acc)

          scheduler.step()

      # if val_acc > best_acc:
      #     best_acc = val_acc
          if val_loss < best_val_loss:
              best_val_loss = val_loss
              best_acc = val_acc
              no_improve = 0
              torch.save(model.state_dict(), save_path)
          else:
              no_improve += 1

          if epoch % 5 == 0 or epoch == 1 or val_acc > best_acc - 0.01:
              print(
              f"Epoch {epoch:3d}/{epochs}  "
              f"train={history['train_loss'][-1]:.4f}  "
              f"val={history['val_loss'][-1]:.4f}  "
              f"acc={val_acc:.3f}"
          )

      # if val_acc >= accuracy_threshold or no_improve >= no_improve_count:
      #     break
          if val_acc >= accuracy_threshold and no_improve >= no_improve_count:
              break

      plt.figure()
      plt.subplot(121)
      plt.title("Loss")
      plt.plot(history["train_loss"], label="train")
      plt.plot(history["val_loss"], label="val")
      plt.subplot(122)
      plt.title("Accuracy")
      plt.plot(history["val_acc"], label="val acc", color="green")
      plt.legend()
      plt.tight_layout()
      plt.savefig(save_dir / "metrics.png")
      plt.show()
  return model

def show_predictions(loader, model, save_dir, n=8):
    model.eval()
    it = iter(loader)
    images, cls_t, bbox_t = next(it)
    images = images.to(device)
    with torch.no_grad():
        cls_pred, bbox_pred = model(images)
    print(bbox_pred[0], bbox_t[0])
    print(bbox_pred[1], bbox_t[1])
    preds = cls_pred.argmax(1).cpu()

    fig, axes = plt.subplots(2, n // 2, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        img_np = images[i].cpu().numpy().transpose(1, 2, 0)
        H, W = img_np.shape[:2]

        cx, cy, bw, bh = bbox_t[i].numpy()
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        ax.add_patch(
            Rectangle(
                (x1, y1),
                bw * W,
                bh * H,
                linewidth=2,
                edgecolor="blue",
                facecolor="none",
            )
        )

        cx, cy, bw, bh = bbox_pred[i].cpu().numpy()
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        ax.add_patch(
            Rectangle(
                (x1, y1),
                bw * W,
                bh * H,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                linestyle="--",
            )
        )

        gt_name = classes[cls_t[i]]
        pr_name = classes[preds[i]]
        color = "green" if preds[i] == cls_t[i] else "red"
        ax.set_title(f"Real:{gt_name}  Predicted:{pr_name}", color=color, fontsize=9)
        ax.imshow(img_np)
        ax.axis("off")

    plt.tight_layout()
    plt.suptitle(save_dir.name)
    plt.savefig(save_dir / "predict.png")
    plt.show()



root = Path("./shapes")
shapes_dataset = Path(root / "shapes_dataset")
shapes_dataset_ds = Path(root / "shapes_dataset_bg")
shapes_dataset_random = Path(root / "shapes_dataset_random")

for p in [shapes_dataset, shapes_dataset_ds, shapes_dataset_random]:
    if not p.exists():
        print(f"ВНИМАНИЕ: Путь {p.absolute()} не найден!")

dataset_train, dataset_val = make_dataset(shapes_dataset)
dataset_ds_train, dataset_ds_val = make_dataset(shapes_dataset_ds)
dataset_random_train, dataset_random_val = make_dataset(shapes_dataset_random)

model_dataset = train(dataset_train, dataset_val, shapes_dataset) 
model_dataset_ds = train(dataset_ds_train, dataset_ds_val, shapes_dataset_ds) 
model_dataset_random = train(dataset_random_train, dataset_random_val, shapes_dataset_random) 

show_predictions(dataset_val, model_dataset, shapes_dataset)
show_predictions(dataset_ds_val, model_dataset_ds, shapes_dataset_ds)
show_predictions(dataset_random_val, model_dataset_random, shapes_dataset_random)
