import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models
import cv2
import time
from PIL import Image
from pathlib import Path
import numpy as np
from collections import deque


class Buffer():
  def __init__(self, maxsize=16):
    self.frames = deque(maxlen=maxsize)
    self.labeles = deque(maxlen=maxsize)

  def append(self, tensor, label):
    self.frames.append(tensor)
    self.labeles.append(label)

  def __len__(self):
    return len(self.frames)
  
  def get_batch(self, device = torch.device("cpu")):
    images = torch.stack(list(self.frames)).to(device)
    labeles = torch.tensor(list(self.labeles), dtype=torch.float32).to(device)
    return images, labeles

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

def build_model(model, device = torch.device("cpu"), model_path = Path('./tmp/model.pth')):
  weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
  model = torchvision.models.efficientnet_b0(weights)
  for param in model.features:
      param.requires_grad = False

  features = model.classifier[-1].in_features
  model.classifier[-1] = nn.Linear(features, 1)
  if model_path.exists():
    model.load_state_dict(torch.load(model_path, map_location=device))
  return model.to(device)

def train(model, optimizer, buffer, device = torch.device("cpu")):
  if len(buffer) < 10:
    return None
  model.train()
  images, labeles = buffer.get_batch(device)
  optimizer.zero_grad()
  predictions = model(images).squeeze(1)
  loss = criterion(predictions, labeles)
  loss.backward()
  optimizer.step()
  return loss.item()

def predict(model, frame, transforms_pipeline, device = torch.device("cpu"), probability = 0.5):
  model.eval()
  tensor = transforms_pipeline(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  tensor = tensor.unsqueeze(0).to(device)
  with torch.no_grad():
    predicted = model(tensor).squeeze()
    prob = torch.sigmoid(predicted).item()
  label = 'person' if prob > probability  else "no_person"
  return label, prob



if __name__ == "__main__":
  # is_active = 0
  # while is_active != 1 or is_active != 2:
  #   print("Выберите метод обучения:")
  #   print("1. Активный")
  #   print("2. Пассвный")
  #   is_active = int(input("Выбирите (1, 2)"))
  
  # if 
  
  device = choose_device()

  alexNet_weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
  efficNet_weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
  alexNet_model = torchvision.models.alexnet(alexNet_weights)
  efficNet_model = torchvision.models.efficientnet_b0(efficNet_weights)

  AlexNet = build_model(alexNet_model, device)
  EfficientNet = build_model(efficNet_model, device)

  criterion = nn.BCEWithLogitsLoss()
  optimizer_AlexNet = torch.optim.Adam(
    filter(lambda p: p.requires_grad, AlexNet.parameters()),
    lr=0.0001
  )
  optimizer_EfficientNet = torch.optim.Adam(
    filter(lambda p: p.requires_grad, EfficientNet.parameters()),
    lr = 0.0001
  )

  transforms_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
  ])

  buffer = Buffer()
  alex_path = Path('./tmp/AlexNet.pth')
  effi_path = Path("./tmp/EfficintNet.pth")
  count_labeled = 0

  cap = cv2.VideoCapture(0)
  cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

  while True:
    _, frame =  cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFf 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if key == ord("q"):
      break
    elif key == ord("1"): #Person
      tensor = transforms_pipeline(image)
      buffer.append(tensor, 1.0)
      count_labeled += 1
    elif key == ord("2"): #No person
      tensor = transforms_pipeline(image)
      buffer.append(tensor, 0.0)
      count_labeled += 1
    elif key == ord("p"): #Preditc
      t = time.perf_counter()
      label_alex, prob_alex = predict(AlexNet, frame, transforms_pipeline, device)
      label_effi, prob_effi = predict(EfficientNet, frame, transforms_pipeline, device)
      print(f"Time: {time}")
      print(f"AlexNet: label = {label_alex}, prob = {prob_alex}")
      print(f"EfficientNet: label = {label_effi}, prob = {prob_effi}")
    elif key == ord("s"): #Save model
      torch.save(AlexNet.state_dict(), alex_path)
      torch.save(EfficientNet.state_dict(), effi_path)

    # print(len(buffer))
    if count_labeled >= buffer.frames.maxlen:
      loss_alex = train(AlexNet, optimizer_AlexNet, buffer, device)
      loss_effi= train(EfficientNet, optimizer_EfficientNet, buffer, device)
      if loss_alex and loss_effi:
        print(f"AlexNet loss = {loss_alex}")
        print(f"EfficientNet loss = {loss_effi}")
      count_labeled = 0

  cap.release()
  cv2.destroyAllWindows()
