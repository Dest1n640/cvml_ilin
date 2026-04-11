from pathlib import Path
import shutil
import random

random.seed(42)

root = Path(r"./dataset")

val_split = 0.2

images_path = root / "images"
labels_path = root / "labels"

out_path = root / "ds"

classes = {k:v for k, v in enumerate((root / "classes.txt").open().read().split())}

print(classes)

pathes = list(labels_path.glob("*.txt"))

len_files = len(pathes) // len(classes)
n_split = int(len_files * (1 - 0.2))

print(n_split)

images = {p.stem: p for p in images_path.glob("*")}


print(images)

counts = {k: 0 for k in classes}

for i, path in enumerate(pathes):
    label = path.open().read().split(" ")[0]
    
    if not label:
        label = "1"
    label = int(label)
    counts[label] += 1
    if counts[label] < n_split:
        save_image_path = out_path / "images" / "train" / classes[label]
        save_label_path = out_path / "labels" / "train" / classes[label]
        save_image_path.mkdir(parents=True, exist_ok=True)
        save_label_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, save_label_path)
        shutil.copy(images[path.stem], save_image_path)
    else:
        print(label)
        save_image_path = out_path / "images" / "val" / classes[label]
        save_label_path = out_path / "labels" / "val" / classes[label]
        save_image_path.mkdir(parents=True, exist_ok=True)
        save_label_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, save_label_path)
        shutil.copy(images[path.stem], save_image_path)

