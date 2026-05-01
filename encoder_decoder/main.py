import torch
import matplotlib.pyplot as plt
from train import Decoder, Encoder, ImageDataset
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

mode_arr = [1, 2, 3, 4]

for mode in mode_arr:
  encoder = Encoder().to(device)
  decoder = Decoder().to(device)

  encoder_path = Path(f"tmp/encoder{mode}.pth")
  decoder_path = Path(f"tmp/decoder{mode}.pth")
  if not encoder_path.exists or not decoder_path.exists:
    print(f"{encoder_path} или {decoder_path} не существует")
    continue

  encoder.load_state_dict(torch.load(encoder_path, map_location=device))
  decoder.load_state_dict(torch.load(decoder_path, map_location=device))

  encoder.eval()
  decoder.eval()

  dataset = ImageDataset(mode=mode, n=10, size=256)
  image, _ = dataset[0]

  with torch.no_grad():
      input_tensor = image.unsqueeze(0).to(device) 
      latent = encoder(input_tensor)
      output = decoder(latent)

  img_orig = image.squeeze().cpu().numpy()
  img_recon = output.squeeze().cpu().numpy()
  diff = torch.abs(torch.tensor(img_orig) - torch.tensor(img_recon)).numpy()
  

  plt.figure(figsize=(15, 5))

  plt.subplot(1, 3, 1)
  plt.title(f"Original (Mode {mode})")
  plt.imshow(img_orig, cmap='gray')
  plt.axis('off')

  plt.subplot(1, 3, 2)
  plt.title("Reconstructed")
  plt.imshow(img_recon, cmap='gray')
  plt.axis('off')

  plt.subplot(1, 3, 3)
  plt.title("Difference (Abs)")
  plt.imshow(diff, cmap='hot')
  plt.colorbar()
  plt.axis('off')

  plt.show()
