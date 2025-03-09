import torch.optim
from torchvision.transforms import ToTensor
from cifar_dataset import CIFARDataset
from models import SimpleNeuralNetwork
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report

if __name__ == '__main__':
  # Số epoch để train mô hình
  num_epochs = 100

  # Tạo dataset và dataloader cho tập train
  train_dataset = CIFARDataset(root="./data", train=True)
  train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=-1,
    drop_last=True
  )

  # Tạo dataset và dataloader cho tập test
  test_dataset = CIFARDataset(root="./data", train=False)
  test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=-1,
    drop_last=False
  )

  # Khởi tạo mô hình mạng neural đơn giản
  model = SimpleNeuralNetwork(num_classes=10)

  # Hàm loss dùng CrossEntropyLoss cho bài toán phân loại
  criterion = nn.CrossEntropyLoss()

  # Sử dụng thuật toán tối ưu SGD với learning rate = 1e-3 và momentum = 0.9
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

  # Số lượng batch trong tập train
  num_iters = len(train_dataloader)

  # Nếu có GPU, chuyển mô hình sang GPU
  if torch.cuda.is_available():
    model.cuda()

  # Bắt đầu vòng lặp huấn luyện
  for epoch in range(num_epochs):
    model.train()  # Đặt mô hình ở chế độ training

    # Lặp qua từng batch trong tập train
    for iter, (images, labels) in enumerate(train_dataloader):
      if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

      # Forward pass: Dự đoán đầu ra
      outputs = model(images)
      loss_value = criterion(outputs, labels)

      # In loss sau mỗi 10 batch
      if iter + 1 % 10:
        print("Epoch {}/{}. Iteration {}/{}. Loss {}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))

      # Backward pass: Tính gradient và cập nhật trọng số
      optimizer.zero_grad()
      loss_value.backward()
      optimizer.step()

    # Đặt mô hình vào chế độ evaluation
    model.eval()
    all_predictions = []
    all_labels = []

    # Lặp qua tập test để đánh giá mô hình
    for iter, (images, labels) in enumerate(test_dataloader):
      all_labels.extend(labels)  # Lưu lại nhãn thực tế
      if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()

      with torch.no_grad():  # Tắt gradient trong quá trình đánh giá
        predictions = model(images)  # Dự đoán (shape: 64x10)
        indices = torch.argmax(predictions.cpu(), dim=1)  # Lấy index có giá trị cao nhất
        all_predictions.extend(indices)  # Lưu lại nhãn dự đoán
        loss_value = criterion(predictions, labels)

    # Chuyển danh sách tensor thành số nguyên
    all_labels = [label.item() for label in all_labels]
    all_predictions = [prediction.item() for prediction in all_predictions]

    # In báo cáo phân loại sau mỗi epoch
    print("Epoch {}".format(epoch+1))
    print(classification_report(all_labels, all_predictions))
