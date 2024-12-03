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
        return z