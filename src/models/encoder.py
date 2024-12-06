from torch import nn
from lightly.models.modules.heads import VICRegProjectionHead


class Encoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = VICRegProjectionHead(
            input_dim=512,
            hidden_dim=1024,
            output_dim=1024,
            num_layers=3,
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return x, z
    

class SimpleCNN(nn.Module):
    def __init__(self, embed_size, input_channel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, 12, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(12, 12, padding=1, kernel_size=3)
        self.conv3 = nn.Conv2d(12, 12, padding=1, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(12)
        self.bn2 = nn.BatchNorm2d(12)
        self.bn3 = nn.BatchNorm2d(12)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d((5, 5), stride=2)
        self.pool2 = nn.MaxPool2d((5, 5), stride=5)
        # h -> (5, 5, stride=1) -> (3, 3)
        # h = 65 -> 8748
        self.fc1 = nn.Linear(432, 4096)
        self.fc2 = nn.Linear(4096, embed_size)

    def forward(self, x):
        # h,w = 65
        x = self.conv1(x)        
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = x2 + x1
        x2 = self.pool1(x2)
        # h,w = 31 

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = x3 + x2
        x3 = self.pool2(x3)
        # h,w = 6

        x3 = x3.view(x3.size(0), -1)
        # (b,12*6*6)
        x3 = self.fc1(x3)
        x3 = self.relu(x3)
        x3 = self.fc2(x3)
        return x3