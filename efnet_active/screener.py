import cv2
from pathlib import Path

image_path = Path("./tmp/images/")
num = 1

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

while True:
  _, frame = cap.read()
  key = cv2.waitKey(1) & 0xFF

  if key == ord("q"):
    break
  if key == ord("s"):
    cv2.imwrite(str(image_path / f"image{num}.png"), frame)
    num += 1

