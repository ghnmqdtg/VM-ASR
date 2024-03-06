import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from base import BaseModel
except:
    # Used for debugging data_loader
    # Add the project root directory to the Python path
    import os
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    # Now you can import BaseModel
    from base.base_model import BaseModel


class MambaUNet(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ToyUNet(BaseModel):
    """
    A simple toy U-Net model that has same shape of input and output.
    """

    def __init__(self, in_channels=1, out_channels=1, scale=2, dropout=0.0, batchnorm=False):
        super().__init__()
        # Define the channel sizes and kernel size for each layer
        channels_down = [in_channels, 32, 64, 128, 128]
        channels_up = [128, 128, 64, 32, out_channels]

        # Activation function
        self.activation = nn.ReLU()

        # Define the downscaling network
        self.down_net = nn.ModuleList()
        for i in range(len(channels_down) - 1):
            down_block = nn.Sequential()
            conv = nn.Conv2d(
                channels_down[i], channels_down[i+1], kernel_size=(3, 3), stride=2, padding=1)
            down_block.append(conv)
            down_block.append(self.activation)
            if batchnorm:
                down_block.append(nn.BatchNorm2d(channels_down[i+1]))
            self.down_net.append(nn.Sequential(*down_block))

        # Define the bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels_down[-1], channels_down[-1],
                      kernel_size=(3, 3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_up[0], channels_up[0] * scale ** 2, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=scale)
        )

        # Define the upscaling network with PixelShuffle and skip connections
        self.up_net = nn.ModuleList()
        for i in range(len(channels_up) - 1):
            up_block = nn.Sequential()
            conv = nn.Conv2d(
                channels_up[i] * scale, channels_up[i+1] * scale ** 2, kernel_size=(3, 3), stride=1, padding=1)
            up_block.append(conv)
            up_block.append(self.activation)
            up_block.append(nn.PixelShuffle(upscale_factor=scale))
            if batchnorm:
                up_block.append(nn.BatchNorm2d(channels_up[i+1]))
            self.up_net.append(nn.Sequential(*up_block))

    def forward(self, x):
        # Residual connection
        y = x.clone()
        # Downscaling path
        down_outputs = []
        for down_block in self.down_net:
            x = down_block(x)
            # print(x.shape)
            down_outputs.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        # print(x.shape)

        # Upscaling path
        for up_block in self.up_net:
            # Concatenate the skip connection
            x = torch.cat((x, down_outputs.pop()), dim=1)
            x = up_block(x)
            # print(x.shape)

        # Add the residual connection
        x += y

        return x


if __name__ == "__main__":
    model = ToyUNet().to('cuda')
    print(model)
    x = torch.rand(128, 1, 512, 128).to('cuda')
    y = model(x)
    print(y.shape)
