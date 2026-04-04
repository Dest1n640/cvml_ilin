import cv2
import time
from train_model import choose_device, predict
from pathlib import Path
import torch
import torchvision

model_path = Path("./tmp/model.pth")
device = choose_device()
model = torchvision.models.efficientnet_b0(weights=None)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

while True:
  _, frame =  cap.read()
  cv2.imshow("Camera", frame)
  key = cv2.waitKey(1) & 0xFf 
  image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  label, prob = predict(model, frame, device)
  cv2.putText(image,
              "Predict - " + {label}, 
              (10, 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              (255, 0, 0), 2)
  print(label, prob)

  if key == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
