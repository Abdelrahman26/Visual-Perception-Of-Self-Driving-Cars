import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias = False), # bias = False cause we will use BatchNorm next
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias = False), # bias = False cause we will use BatchNorm next
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace = True)
        )

    def forward(self, x_img):
        return self.conv(x_img)
