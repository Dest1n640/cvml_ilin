from train_model import CyrrilicCNN, choose_device, build_dataloaders
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


device = choose_device()

save_path = Path("./tmp")
model_path = save_path / 'model.pth'
model = CyrrilicCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

path = Path("./tmp/Cyrillic/")
_, _, test_loader, test_data = build_dataloaders(path, batch_size=16)

with torch.no_grad():
  for i in range(10):
    random_num = np.random.randint(0, len(test_data))
    image, label = test_data[random_num]
    output = model(image.unsqueeze(0).to(device))
    _, predicted = torch.max(output.data, 1)


    print(f"Истинная метка = {label}")
    print(f"Предсказание модели = {predicted[0]}")

    plt.imshow(image.squeeze(0))
    plt.title(f"true={label}, prediction={predicted[0]}")
    plt.show()

total = 0.0
correct = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test accuracy: {100 * correct / total}')

train_png = Path("train.png")
img = plt.imread(train_png)
plt.title("Графики обучения модели")
plt.imshow(img)
plt.show()
