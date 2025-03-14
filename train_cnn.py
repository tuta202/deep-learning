import multiprocessing
import torch.optim
from torchvision.transforms import ToTensor
from cifar_dataset import CIFARDataset
from models import SimpleCNN
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(writer, cm, class_names, epoch):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """

  figure = plt.figure(figsize=(20, 20))
  # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
  plt.imshow(cm, interpolation='nearest', cmap="ocean")
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.

  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      color = "white" if cm[i, j] > threshold else "black"
      plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  writer.add_figure('confusion_matrix', figure, epoch)
    
def get_args():
  parser = ArgumentParser(description="CNN training")
  parser.add_argument("--root", "-r", type=str, default="./data", help="Root of the dataset")
  parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
  parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")
  parser.add_argument("--image-size", "-i", type=int, default=32, help="Image size")
  parser.add_argument("--logging", "-l", type=str, default="tensorboard")
  parser.add_argument("--trained_models", "-t", type=str, default="trained_models")
  parser.add_argument("--checkpoint", "-c", type=str, default=None)
  args = parser.parse_args()
  return args
  
if __name__ == '__main__':
  args = get_args()
  
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu") 
    
  num_workers = int(multiprocessing.cpu_count() / 2)
  print(num_workers)
  
  num_epochs = 100

  train_dataset = CIFARDataset(root=args.root, train=True)
  train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True
  )

  test_dataset = CIFARDataset(root=args.root, train=False)
  test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False
  )

  if os.path.isdir(args.logging):
    shutil.rmtree(args.logging)
  if not os.path.isdir(args.trained_models):
    os.mkdir(args.trained_models)
    
  writer = SummaryWriter(args.logging)

  model = SimpleCNN(num_classes=10).to(device)
  
  criterion = nn.CrossEntropyLoss()

  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

  if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    best_acc = checkpoint["best_acc"]
  else:
    start_epoch = 0
    best_acc = 0
    
  num_iters = len(train_dataloader)

  for epoch in range(start_epoch, args.epochs):
    model.train()  
    progress_bar = tqdm(train_dataloader)

    for iter, (images, labels) in enumerate(progress_bar):
      images = images.to(device)
      labels = labels.to(device)

      outputs = model(images)
      loss_value = criterion(outputs, labels)

      progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iters, loss_value))
      writer.add_scalar("Train/Loss", loss_value, epoch*num_iters+iter)
      
      optimizer.zero_grad()
      loss_value.backward()
      optimizer.step()

    model.eval()
    all_predictions = []
    all_labels = []

    for iter, (images, labels) in enumerate(test_dataloader):
      all_labels.extend(labels)  
      images = images.to(device)
      labels = labels.to(device)

      with torch.no_grad():  
        predictions = model(images) 
        indices = torch.argmax(predictions.cpu(), dim=1) 
        all_predictions.extend(indices)  
        loss_value = criterion(predictions, labels)

    all_labels = [label.item() for label in all_labels]
    all_predictions = [prediction.item() for prediction in all_predictions]
    plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), class_names=test_dataset.categories, epoch=epoch)

    print("best_acc", best_acc)
    accuracy = accuracy_score(all_labels, all_predictions)
    print("Epoch {}: Accuracy: {}".format(epoch+1, accuracy))
    writer.add_scalar("Val/Accuracy", accuracy, epoch)
    # torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_models))
    checkpoint = {
      "epoch": epoch+1,
      "best_acc": best_acc,
      "model": model.state_dict(), # W của model
      "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
    if accuracy > best_acc:
      checkpoint = {
        "epoch": epoch + 1,
        "best_acc": best_acc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
      }
      torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
      best_acc = accuracy
    # print(classification_report(all_labels, all_predictions))