import torch
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)

image = torch.rand(2, 3, 224, 224)

output = model(image)
