from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from animal_dataset import AnimalDataset
from cifar_dataset import CIFARDataset
import multiprocessing

if __name__ == '__main__':
  num_workers = int(multiprocessing.cpu_count() / 2)

  transform = Compose([
    Resize((200,200)),
    ToTensor()
  ])
  training_data = AnimalDataset(root="data", train=True, transform=transform)
  # training_data = CIFARDataset(root="data", train=True)
  image, label = training_data.__getitem__(12345)

  training_dataloader = DataLoader(
    dataset=training_data,
    batch_size=1000,
    num_workers=num_workers,
    shuffle=True,
    drop_last=False, 
  )

  for images, labels in training_dataloader:
    print('shape', images.shape)
    print('type', type(images))
    print('labels', labels)
