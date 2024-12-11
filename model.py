import torch
import torch.nn as nn


class SymbolCNN(nn.Module):

    def __init__(self, num_classes, image_size, pre_crop_size=(500, 300)):
        super(SymbolCNN, self).__init__()
        self.image_size = image_size
        self.pre_crop_size = pre_crop_size

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.33)

        # Calculate flattened size dynamically
        x = torch.randn(1, 1, image_size[0], image_size[1])
        x = self._forward_conv(x)
        conv_out_shape = x.shape[1:]  # Shape after convolutions

        # Fully connected layers that take both flattened features and w/h
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_out_shape[0] * conv_out_shape[1] * conv_out_shape[2] + 1, 64)
        # +2 for width and height
        self.bn4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)

    def _forward_conv(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, x, sizes):
        x = self._forward_conv(x)
        x = self.flatten(x)

        # Normalize width and height before concatenating
        sizes = sizes.float() / max(self.pre_crop_size)  # Normalize width

        # Concatenate flattened convolutional features and width/height
        x = torch.cat((x, sizes.unsqueeze(1)), dim=1)

        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
