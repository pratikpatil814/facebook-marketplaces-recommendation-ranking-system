import torch
from torchvision.io import read_image
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class facbook_RN_50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.resnet50.fc = torch.nn.Linear(2048,13)

    def forward(self,x):
        return F.sigmoid(self.resnet50(x))
        