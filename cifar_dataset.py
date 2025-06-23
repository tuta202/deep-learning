import torch
from torch.utils.data import Dataset
import os
import pickle
import cv2
import numpy as np

class CIFARDataset(Dataset):
  def __init__(self, root="data", train=True):
    self.categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    root = os.path.join(root, "cifar-10-batches-py")
    if train:
      data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
    else:
      data_files = [os.path.join(root, "test_batch")]

    self.images = []
    self.labels = []
    for data_file in data_files:
      with open(data_file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        self.images.extend(dict[b'data'])
        self.labels.extend(dict[b'labels'])

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    image = np.reshape(self.images[idx], (3, 32, 32)).astype(np.float32)
    label = self.labels[idx]
    return image/255., label

if __name__ == '__main__':
  dataset = CIFARDataset(root="data", train=True)
  image, label = dataset.__getitem__(100)
  image = np.transpose(image, (1, 2, 0))
  cv2.imshow("image", cv2.resize(image, (320, 320)))
  cv2.waitKey(0)