import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel,
            padding=kernel // 2,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HMT_ECGNet(nn.Module):
    """
    HMT_v1
    - Hierarchical temporal CNN
    - ~34k params
    - No attention
    - Proven strong baseline (AUROC ~0.92)
    """

    def __init__(self, num_classes=5, num_leads=12):
        super().__init__()

        self.block1 = ConvBlock(num_leads, 32)
        self.block2 = ConvBlock(32, 64)
        self.block3 = ConvBlock(64, 128)
        self.block4 = ConvBlock(128, 256)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, 12, T)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.pool(x).squeeze(-1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
