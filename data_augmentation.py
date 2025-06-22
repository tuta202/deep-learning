from animal_dataset import AnimalDataset
from torchvision.transforms import Compose, Resize, ToTensor, RandomAffine, ColorJitter
import torch
import cv2
import numpy as np

if __name__ == '__main__':
  train_transform = Compose([
    RandomAffine(
      degrees=(-15, 15),
      translate=(0.05, 0.05),
      scale=(0.85, 1.15),
      shear=5
    ),
    Resize((224, 224)),
    ColorJitter(
      brightness=0.5,
      contrast=0.5,
      saturation=0.25,
      hue=0.5
    ),
    ToTensor()
  ])

  train_dataset = AnimalDataset(root="./data", train=True, transform=train_transform)

  image, _ = train_dataset.__getitem__(12345)

  print('image type: ', type(image))
  image = (torch.permute(image, dims=(1, 2, 0)) * 255.).numpy().astype(np.uint8)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  cv2.imshow("test image", image)
  cv2.waitKey(0)
