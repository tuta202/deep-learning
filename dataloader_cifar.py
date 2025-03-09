from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from animal_dataset import AnimalDataset

if __name__ == '__main__':
  transform = Compose([
    Resize((200,200)),
    ToTensor()
  ])
  training_data = AnimalDataset(root="./data", train=True, transform=transform)
  # training_data = CIFAR10(root="data", train=True, transform=ToTensor())
  image, label = training_data.__getitem__(12345)

  training_dataloader = DataLoader(
    dataset=training_data,
    batch_size=1000,
    num_workers=-1,
    shuffle=True,
    drop_last=False, 
  )

  for images, labels in training_dataloader:
    print(images.shape)
    print(labels)
