from unet_road import UNet, RoadsDataset

from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

def main():
  dataset_path = Path("./roads")

  all_images = sorted(list((dataset_path / "images").glob("*.png")))
  all_masks = sorted(list((dataset_path / "masks").glob("*.png")))

  _, test_imgs, _, test_masks = train_test_split(
    all_images, all_masks, test_size=0.2
  )

  test_dataset = RoadsDataset(test_imgs, test_masks, is_train=False)
  test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

  device = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.mps.is_available()
                        else "cpu")

  unet_path = Path("./tmp/unet.pth")

  unet = UNet()
  state_dict = torch.load(unet_path, map_location=device)
  if unet_path.exists():
    state_dict = torch.load(unet_path, map_location=device)
    unet.load_state_dict(state_dict)
  else:
    raise ValueError()
  unet.to(device)
  unet.eval()

  images_to_show = len(test_dataset)
  current_image = 0

  for image, mask in test_dataloader:
    image = image.to(device)
    mask = mask.to(device)

    with torch.no_grad():
      pred_logits = unet(image)
      pred_probs = torch.sigmoid(pred_logits)
      pred_mask = (pred_probs > 0.5).float()

    img_vis = image[0].cpu().permute(1, 2, 0).numpy()
    mask_vis = mask[0, 0].cpu().numpy()
    pred_vis = pred_mask[0, 0].cpu().numpy()
    difference = np.abs(mask_vis - pred_vis)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[1].imshow(mask_vis)
    axes[0].imshow(img_vis)
    axes[0].set_title("Исходное изображение")
    axes[0].axis('off')

    axes[1].imshow(mask_vis, cmap='gray')
    axes[1].set_title("Истинная маска")
    axes[1].axis('off')
    
    axes[2].imshow(pred_vis, cmap='gray')
    axes[2].set_title("Предсказание")
    axes[2].axis('off')
    
    axes[3].imshow(difference, cmap='hot')
    axes[3].set_title("Разница")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    current_image += 1
    if current_image >= images_to_show:
      break

if __name__ == "__main__":
  main() 
