import cv2
import time
from train_model import choose_device, predict
from torchvision import transforms
from pathlib import Path
import torch
import torchvision

transforms_pipeline = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])


alex_path = Path("./tmp/AlexNet.pth")
effi_path = Path("./tmp/EfficientNet.pth")
device = choose_device()
AlexNet = torchvision.models.alexnet(weights = None)
EfficienNet = torchvision.models.efficientnet_b0(weights = None)
AlexNet.classifier[-1] = torch.nn.Linear(AlexNet.classifier[-1].in_features, 1)
EfficienNet.classifier[-1] = torch.nn.Linear(EfficienNet.classifier[-1].in_features, 1)
AlexNet.to(device)
EfficienNet.to(device)
AlexNet.load_state_dict(torch.load(alex_path, map_location=device))
EfficienNet.load_state_dict(torch.load(effi_path, map_location=device))
AlexNet.eval()
EfficienNet.eval()

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

while True:
  _, frame = cap.read()
  key = cv2.waitKey(1) & 0xFf
  alex_label, alex_prob = predict(AlexNet, frame, transforms_pipeline, device)
  effi_label, effi_prob = predict(EfficienNet, frame, transforms_pipeline, device)

  cv2.putText(frame,
              "AlexNet predict - " + alex_label,
              (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.7,
              (255, 0, 0),
              2)

  cv2.putText(frame,
              "EfficienNet predict - " + effi_label,
              (10, 60),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.7,
              (0, 255, 0),
              2)

  cv2.imshow("Camera", frame)

  print(f"AlexNet: label = {alex_label}, prob = {alex_prob}")
  print(f"EfficientNet: label = {effi_label}, prob = {effi_prob}")

  if key == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
