import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding="same")
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding="valid")
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, padding="valid")
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.tanh = nn.Tanh()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(480, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, image):
        out = self.tanh(self.conv1(image))
        out = self.pool(out)
        out = self.tanh(self.conv2(out))
        out = self.pool(out)
        out = self.tanh(self.conv3(out))
        out = self.flat(out)
        out = self.tanh(self.fc1(out))
        out = self.fc2(out)

        return out
