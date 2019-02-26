import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

class CategoryNet(nn.Module):
    def __init__(self, numCategories):
        super(CategoryNet, self).__init__()
        self.numCategories = numCategories;
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.mp1 = nn.MaxPool2d(kernel_size=3)
        self.dp1 = nn.Dropout2d(0.25)

        self.conv2_1 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.relu2_2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.mp2 = nn.MaxPool2d(kernel_size=3)
        self.dp2 = nn.Dropout2d(0.25)

        self.conv3_1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3)
        self.relu3_2 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d(kernel_size=2)
        self.dp3 = nn.Dropout2d(0.25)

        self.ln4_1 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(256)
        self.bp4 = nn.Dropout2d(0.5)
        self.ln4_2 = nn.Linear(256, self.numCategories);

    def forward(self,x):
        x = self.dp1(self.mp1(self.bn1(self.relu1(self.conv1(x)))))
        x = self.dp2(self.mp2(self.bn2(self.relu2_2(self.conv2_2(self.relu2_1(self.conv2_1(x)))))))
        x = self.dp3(self.mp3(self.bn3(self.relu3_2(self.conv3_2(self.relu3_1(self.conv3_1(x)))))))
        x = x.view(x.size(0), -1)
        x = self.ln4_1(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.bp4(self.bn4(self.relu4(x)))
        x = x.view(x.size(0), -1)
        x = self.ln4_2(x)
        return x


class ColorNet(nn.Module):
    def __init__(self, numColors):
        super(ColorNet, self).__init__()
        self.numColors = numColors;
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.mp1 = nn.MaxPool2d(kernel_size=3)
        self.bp1 = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.bp2 = nn.Dropout2d(0.25)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.mp3 = nn.MaxPool2d(kernel_size=2)
        self.bp3 = nn.Dropout2d(0.25)

        self.ln4_1 = nn.Linear(1152, 128)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(128)
        self.bp4 = nn.Dropout2d(0.5)
        self.ln4_2 = nn.Linear(128, self.numColors)

    def forward(self,x):
        x = self.bp1(self.mp1(self.bn1(self.relu1(self.conv1(x)))))
        x = self.bp2(self.mp2(self.bn2(self.relu2(self.conv2(x)))))
        x = self.bp3(self.mp3(self.bn3(self.relu3(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        x = self.ln4_1(x)
        x = x.view(x.size(0), -1, 1, 1)
        x = self.bp4(self.bn4(self.relu4(x)))
        x = x.view(x.size(0), -1)
        x = self.ln4_2(x)
        return x
