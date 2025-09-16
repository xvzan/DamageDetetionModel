import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out += residual
        out = self.silu(out)
        return out


class MyModel(torch.nn.Module):
    def __init__(
        self,
        input_channel=8,
        input_dim=128,
        num_classes=4,
        # dropout=0.1,
    ):
        super(MyModel, self).__init__()
        self.input_conv1 = nn.Conv1d(
            in_channels=input_channel,
            out_channels=input_dim,
            kernel_size=33,
            stride=1,
            padding=16,
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(input_dim, kernel_size=9, dilation=1),
            ResidualBlock(input_dim, kernel_size=9, dilation=2),
            ResidualBlock(input_dim, kernel_size=9, dilation=4),
            ResidualBlock(input_dim, kernel_size=9, dilation=8),
            ResidualBlock(input_dim, kernel_size=9, dilation=4),
            ResidualBlock(input_dim, kernel_size=9, dilation=2),
            ResidualBlock(input_dim, kernel_size=9, dilation=1),
            ResidualBlock(input_dim, kernel_size=9, dilation=2),
            ResidualBlock(input_dim, kernel_size=9, dilation=4),
            ResidualBlock(input_dim, kernel_size=9, dilation=8),
            ResidualBlock(input_dim, kernel_size=9, dilation=4),
            ResidualBlock(input_dim, kernel_size=9, dilation=2),
            ResidualBlock(input_dim, kernel_size=9, dilation=1),
            ResidualBlock(input_dim, kernel_size=9, dilation=2),
            ResidualBlock(input_dim, kernel_size=9, dilation=4),
            ResidualBlock(input_dim, kernel_size=9, dilation=8),
            ResidualBlock(input_dim, kernel_size=9, dilation=4),
            ResidualBlock(input_dim, kernel_size=9, dilation=2),
            ResidualBlock(input_dim, kernel_size=9, dilation=1),
        )
        self.output_fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.input_conv1(x)
        x = self.res_blocks(x)
        # x = self.input_conv2(x)
        num = x.size(-1) // 2
        x = x[:, :, num]
        x = self.output_fc(x)
        return x
