from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound
from collections import deque
import cv2
import time
import numpy as np

def is_person_detected(results, counter, empty_start_time, threshold = 10):
  person_detected = len(results[0]) > 0

  if not person_detected:
    if empty_start_time is None:
      empty_start_time = time.time()
    else:
      elapsed = time.time() - empty_start_time
      if elapsed >= threshold:
        counter = 0
        empty_start_time = time.time()
  else:
    if empty_start_time is not None:
      empty_start_time = None
  
  return counter, empty_start_time


def get_angle(a, b, c):
  cb = np.atan2(c[1] - b[1], c[0] - b[0])
  ab = np.atan2(a[1] - b[1], a[0] - b[0])
  angle = np.rad2deg(cb - ab)
  angle = angle + 360 if angle < 0 else angle
  return 360 - angle if angle > 180 else angle

def is_point_visible(point, min_conf=0.5):
    return point[0] != 0.0 and point[1] != 0.0 and len(point) >= 3 and point[2] > min_conf

def detect_push_up(annotated, keypoints, counter, stage_up, angle_history):
  left_shoulder = keypoints[5]
  right_shoulder = keypoints[6]
  left_elbow = keypoints[7]
  right_elbow = keypoints[8]
  left_wrist = keypoints[9]
  right_wrist = keypoints[10]
  angles =[]

  if is_point_visible(left_shoulder) and is_point_visible(left_elbow) and is_point_visible(left_wrist):
    angles.append(get_angle(left_shoulder, left_elbow, left_wrist))
        
  if is_point_visible(right_shoulder) and is_point_visible(right_elbow) and is_point_visible(right_wrist):
    angles.append(get_angle(right_shoulder, right_elbow, right_wrist))

  if angles:
    avg_angle = sum(angles) / len(angles)
    angle_history.append(avg_angle)
    smoothed_angle = sum(angle_history) / len(angle_history)

    if smoothed_angle < 90 and stage_up:
      stage_up = False

    if smoothed_angle > 160 and not stage_up:
      stage_up = True
      counter += 1

  status = "UP" if stage_up else "DOWN"
  cv2.putText(annotated,
              f"Push ups: {counter} | {status}",
              (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX,
              1.5,
              (0, 255, 0),
              3)

  return counter, stage_up
      

model = YOLO("./yolo26n-pose.pt")

camera = cv2.VideoCapture(0)
ps = None
counter = 0
stage_up = True
empty_start_time = None
prev_counter = 0
angle_history = deque(maxlen=5)

while camera.isOpened():
  ret, frame = camera.read()
  cv2.imshow("Camera", frame)
  key = cv2.waitKey(10) & 0xFF
  if key == ord("q"):
    break

  t = time.perf_counter()
  results = model(frame, verbose=False)
  # print(f"FPS {1 / (time.perf_counter() - t):.1f})")

  prev_counter = counter
  counter, empty_start_time = is_person_detected(results, counter, empty_start_time)

  result = results[0]
  if len(result) > 0 and result.keypoints is not None and len(result.keypoints.xy) > 0:
      keypoints = result.keypoints.data[0].tolist()

      annotator = Annotator(frame)
      annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
      annotated = annotator.result()

      counter, stage_up = detect_push_up(annotated, keypoints, counter, stage_up, angle_history)
      cv2.imshow("POSE", annotated)
  else:
      angle_history.clear()
      cv2.putText(frame,
                  f"Person not in frame. Push ups: {counter}",
                  (10, 30),
                  cv2.FONT_HERSHEY_COMPLEX,
                  1.5,
                  (0, 255, 0),
                  3)
      cv2.imshow("POSE", frame)
  if counter > prev_counter:
      if ps is None:
          ps = playsound("./music.mp3", block=False)
      else:
          if not ps.is_alive():
            ps = None

camera.release()
cv2.destroyAllWindows()
