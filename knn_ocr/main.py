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
              props[0].euler_number]

  return np.array(features, dtype="f4")

def make_train(path):
  symbols = {}
  train = []
  responses = []
  ncls = 0
  for cls in sorted(path.glob("*")):
#     print(cls)
    ncls += 1
    if cls.name[0] == "s":
      symbols[ncls] = cls.name[1]
    else:
      symbols[ncls] = cls.name
    for p in cls.glob("*.png"):
#       print(p)
      train.append(extractor(imread(p)))
      responses.append(ncls)
  train = np.array(train, dtype = "f4").reshape(-1, 4)
  responses = np.array(responses, dtype = 'f4').reshape(-1, 1)
  return train, responses, symbols

data = Path("./task")
for i in range(0, 7):
  image = imread(data / f'{i}.png')

  train, responses, symbols = make_train(data / "train") 
  knn = cv2.ml.KNearest.create()
  knn.train(train, cv2.ml.ROW_SAMPLE, responses)

  gray = image.mean(2)
  binary = gray > 4
  lb = label(binary)
  props = regionprops(lb)

  props.sort(key=lambda x: x.centroid[1]) 

  find = []
  for prop in props: 
    find.append(extractor(prop.image))

  find = np.array(find, dtype = 'f4').reshape(-1, 4)

  ret, result, neighbours, dist = knn.findNearest(find,  4)
  # print(ret, result, neighbours, dist)
  # amount = Counter(result.flatten())
  # print(amount)

  letters = []
  prev_centroid_x = None
  for idx, prop in zip(result.flatten(), props):
      if prev_centroid_x is not None and prop.centroid[1] - prev_centroid_x > 92:
          letters.append(" ")
      letters.append(symbols.get(int(idx)))
      prev_centroid_x = prop.centroid[1]

  full_words = "".join(map(str, letters))

  print(full_words)
  # print(train, response)
  # print(extractor(image))
  # plt.imshow(image)
  # plt.show()
