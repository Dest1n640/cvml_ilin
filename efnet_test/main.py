import torch
from sklearn.metrics import confusion_matrix
from train_model import choose_device, val_loader, val_ds
from pathlib import Path
import torchvision

def model_load(model, path_weight, device = torch.device("cpu")):
  num_feature = model.classifier[1].in_features
  model.classifier[1] = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(num_feature, 3)
  )

  if path_weight.exists():
    model.load_state_dict(torch.load(path_weight, map_location=device))
  model.to(device)
  model.eval()
  return model

def build_confusion_matrix(model, dataset, device=torch.device("cpu")):
  labels_true = []
  labels_predict = []
  with torch.no_grad():
    for images, labels in dataset:
      images = images.to(device)
      output = model(images)
      predict = torch.argmax(output, dim = 1)
      labels_true.extend(labels.cpu().numpy())
      labels_predict.extend(predict.cpu().numpy())
  cm = confusion_matrix(labels_true, labels_predict)
  print(cm)
  return cm

if __name__ == "__main__":
  EffNet0_weight = Path("./tmp/EffNet0.pth")
  EffNet1_weight = Path("./tmp/EffNet1.pth")
  EffNet2_weight = Path("./tmp/EffNet2.pth")

  device = choose_device()

  EffNet0_model = torchvision.models.efficientnet_b0(weights = None)
  EffNet1_model = torchvision.models.efficientnet_b1(weights = None)
  EffNet2_model = torchvision.models.efficientnet_b2(weights = None)

  EffNet0 = model_load(EffNet0_model, EffNet0_weight, device)
  EffNet1 = model_load(EffNet1_model, EffNet1_weight, device)
  EffNet2 = model_load(EffNet2_model, EffNet2_weight, device)

  models = [(EffNet0, "B0"), (EffNet1, "B1"), (EffNet2, "B2")]

  for m, name in models:
    print(f"\nModel {name}:")
    cm = build_confusion_matrix(m, val_loader, device)
    print("-" * 50)
  print(val_ds.class_to_idx)
