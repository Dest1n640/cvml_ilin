from ultralytics import YOLO
from pathlib import Path
import yaml

classes = {0: "cube", 1: "neither", 2: "sphere"}

root = Path("./spheres_and_cubes_new").resolve()

config = {
  "path": str(root.absolute()),
  "train": str((root / "images/train").absolute()),
  "val": str((root / "images/val").absolute()),
  
  "nc": len(classes),
  "names": classes
}

with open(root/"dataset.yaml", "w") as f:
  yaml.dump(config, f, allow_unicode=True)

size = "n" #n, s, m, l, x
model = YOLO(f"yolo26{size}.pt")

data_path = Path(root / "dataset.yaml")

result = model.train(data=data_path,
            epochs=50,
            batch=16,
            workers=4,
            device="mps",
            patience=5,
            optimizer="AdamW",
            lr0=0.001,

            #Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            degrees=5.0,
            scale=0.5,
            translate=0.1,

            conf=0.001,
            iou=0.7,

            project=f"{root}/figures",
            name="yolo26",
            save=True,
            save_period=3,

            verbose = True,
            plots=True,
            val=True,
            close_mosaic=8,
            amp=True, #FP16

            warmup_epochs=5,
            cos_lr=True,
            dropout=0.2,

            imgsz=640
            )

print("DONE")
print(result.save_dir)
