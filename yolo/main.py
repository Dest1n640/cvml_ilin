from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

def make_predict(model, image_path, title):
  result = model.predict(source = image_path, device="mps",
                       conf=0.25, iou=0.45, imgsz=640)[0]

  img = np.array(Image.open(image_path).convert("RGB"))
  plt.imshow(img)
  plt.title(title)

  boxes = result.boxes.xyxy.cpu().numpy()
  cls = result.boxes.cls.cpu().numpy()
  scores = result.boxes.conf.cpu().numpy()

  for box, label, score in zip(boxes, cls, scores):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
      (x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none', edgecolor="red"
    )
    plt.gca().add_patch(rect)
    plt.gca().text(x1, y1 - 10, f"{score:.2f}", color="white", fontsize=12)
  return result.boxes

classes = {0: "cube", 1: "neither", 2: "sphere"}

image_path_sphere = Path("./spheres_and_cubes_new/images/val/sphere/837537a5-IMG20260317111338.jpg")
image_path_neither = Path("./spheres_and_cubes_new/images/val/neither/80aa16d6-dT323g0lV6TDPG5uB_aqJdVGYGlUDMuOAmaIWkcZbwnsLJafy1aZvLSbx5sLZkTCX9hZZC_5PGolwo.jpg")
image_path_cube = Path("./spheres_and_cubes_new/images/val/cube/8220b248-photo_9_2026-03-17_11-31-24.jpg")

model = YOLO('./yolo26n.pt')

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
result_cube = make_predict(model, image_path_cube, "Cube")
plt.subplot(1, 3, 2)
result_sphere = make_predict(model, image_path_sphere, "Sphere")
plt.subplot(1, 3, 3)
result_neither = make_predict(model, image_path_neither, "Neither")
plt.tight_layout()
plt.show()
