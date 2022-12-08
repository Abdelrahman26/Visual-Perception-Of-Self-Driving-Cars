import torch
import torch.nn as nn


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], ):
        super().__init__()
        self.ups = nn.ModuleList()  # Holds submodules in a list
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # down part of UNET
        for feature in features:
            self.downs.append((in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self.ups.append(
                (
                    feature * 2,
                    feature
                )
            )
        # End Of Loop

        # bottleneck of UNET
        self.bottleneck = (
            features[-1],
            features[-1] * 2
        )

        # Final Conv (last layer)
        self.final_conv = nn.Conv2d(
            features[0],
            out_channels,
            kernel_size=1
        )

    # Forward
    def forward(self, x_img):
        # Skip Connection
        skip_connections = []
        # Down
        for down in self.downs:
            x_img = down(x_img)
            skip_connections.append(x_img)
            x_img = self.pool(x_img)

        x_img = self.bottleneck(x_img)
        # reverse skip connection to be from down to up
        skip_connections = skip_connections[::-1]
        # Up
        for idx in range(0, len(self.ups), 2):  # Up Then Double Conv
            x_img = self.ups[idx](x_img)
            skip_connection = skip_connections[idx // 2]
            if x_img.shape != skip_connection.shape:
                x_img = TF.resize(x_img, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x_img), dim=1)
            # Double Conv
            x_img = self.ups[idx + 1](concat_skip)

        # Final Conv, (1*1) convolution
        return self.final_conv(x_img)