from ultralytics import YOLO
from pathlib import Path
from PIL import Image
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import cv2

yolo_path = Path("./spheres_and_cubes_new/figures/yolo26/weights/best.pt")
classes = {0: "cube", 1: "neither", 2: "sphere"}
yolo = YOLO(yolo_path)

camera = cv2.VideoCapture(0)

while True:
  ret, frame = camera.read()
  key = cv2.waitKey(10) & 0xFF
  if key == ord("q"):
    break

  results = yolo.predict(source = frame, device="mps",
                       conf=0.25, iou=0.45, imgsz=640)[0]
  boxes = results.boxes

  for i in range(len(boxes)):
      x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
      conf = float(boxes.conf[i])
      cls_id = int(boxes.cls[i])
      label = classes.get(cls_id, f"ID {cls_id}")
      display_text = f"{label} {conf:.2f}"
      cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(frame, display_text, (x1, y1 - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  cv2.imshow("YOLO", frame)



camera.release()
cv2.destroyAllWindows()
