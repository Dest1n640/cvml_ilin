import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread
from collections import Counter

def extractor(image):
  if image.ndim == 2:
    binary = image
  else:
    gray = np.mean(image, 2).astype("u1")
    threshold = 4
    binary = gray > threshold

#  plt.imshow(binary)
#  plt.show()
  lb = label(binary)
  props = regionprops(lb)

  features = [props[0].eccentricity,
              props[0].solidity,
              props[0].extent,
              props[0].euler_number,
              *props[0].moments_hu]

  return np.array(features, dtype="f4")

def make_train(path):
  symbols = {}
  train = []
  responses = []
  ncls = 0
  for cls in sorted(path.glob("*")):
    ncls += 1
    if cls.name[0] == "s":
      symbols[ncls] = cls.name[1]
    else:
      symbols[ncls] = cls.name
    for p in cls.glob("*.png"):
      train.append(extractor(imread(p)))
      responses.append(ncls)
  train = np.array(train, dtype = "f4").reshape(-1, 11)
  responses = np.array(responses, dtype = 'f4').reshape(-1, 1)
  return train, responses, symbols

data = Path("./task")
for i, _ in enumerate(sorted(data.glob('*.png'))):
  image = imread(data / f'{i}.png')

  train, responses, symbols = make_train(data / "train") 
  knn = cv2.ml.KNearest.create()
  knn.train(train, cv2.ml.ROW_SAMPLE, responses)

  gray = image.mean(2)
  binary = gray > 4
#  binary = binary.astype(np.uint8)
#  kernel = np.ones((5, 5))
#  binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
  lb = label(binary)
  props = regionprops(lb)

  props.sort(key=lambda x: x.centroid[1]) 

  combined_regions = []
  x_threshold = 10

  for region in props:
    top, left, bottom, right = region.bbox
    center_y, center_x = region.centroid

    if combined_regions:
      prev_top, prev_left, prev_bottom, prev_right, prev_center_x = combined_regions[-1]
      if abs(center_x - prev_center_x) < x_threshold:
        combined_regions[-1] = (
          min(top, prev_top), min(left, prev_left), max(bottom, prev_bottom), max(right, prev_right),
          prev_center_x
        )
        continue
    combined_regions.append((top, left, bottom, right, center_x))

  find = []

  for y1, x1, y2, x2, cx in combined_regions:
      region_img = binary[int(y1):int(y2), int(x1):int(x2)]
      find.append(extractor(region_img))

  find = np.array(find, dtype = 'f4').reshape(-1, 11)



  ret, result, neighbours, dist = knn.findNearest(find,  4)
  # print(ret, result, neighbours, dist)
  # amount = Counter(result.flatten())
  # print(amount)


  threshold_x = 30
  letters = []
  prev_right = None
  for idx, box in zip(result.flatten(), combined_regions):
    left = box[1]
    right = box[3]
    if prev_right is not None and left - prev_right > threshold_x:
      letters.append(" ")
    letters.append(symbols.get(int(idx)))
    prev_right = right

  full_words = "".join(map(str, letters))

  print(i, full_words)
