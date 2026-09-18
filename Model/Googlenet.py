import torch
import torch.nn as nn


class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.conv = nn.Sequential(nn.Conv2d(in_channels, 128, kernel_size=1), nn.ReLU())

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.fc(x)
        return x


class Inception(nn.Module):
    def __init__(
        # 输入通道 ，1的输出层，3的一次输出，3的1输出
        self,
        in_channels,
        ch1x1,
        ch3x3_reduce,
        ch3x3,
        ch5x5_reduce,
        ch5x5,
        pool_proj,
    ):
        super().__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, ch1x1, 1), nn.ReLU())

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, 1),
            nn.ReLU(),
            nn.Conv2d(ch3x3_reduce, ch3x3, 3, padding=1),
            nn.ReLU(),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, 1),
            nn.ReLU(),
            nn.Conv2d(ch5x5_reduce, ch5x5, 5, padding=2),
            nn.ReLU(),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        return torch.cat([b1, b2, b3, b4], dim=1)


class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.aux1 = AuxClassifier(512, 10)
        self.aux2 = AuxClassifier(528, 10)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 10)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # 输入来自 3b 后的 MaxPool
        # 480 channels

        self.inception4a = Inception(
            480,
            192,  # 1x1
            96,
            208,  # 1x1 reduce -> 3x3
            16,
            48,  # 1x1 reduce -> 5x5
            64,  # pool -> 1x1
        )
        # 输出: 192 + 208 + 48 + 64 = 512

        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # 输出: 160 + 224 + 64 + 64 = 512

        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # 输出: 128 + 256 + 64 + 64 = 512

        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        # 输出: 112 + 288 + 64 + 64 = 528

        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        # 输出: 256 + 320 + 128 + 128 = 832
        self.inception5a = Inception(
            832, 256, 160, 320, 32, 128, 128
        )  # out = 256 + 320 + 128 + 128 = 832

        self.inception5b = Inception(
            832, 384, 192, 384, 48, 128, 128
        )  # out = 384 + 384 + 128 + 128 = 1024

    def forward(self, x):

        x = self.stem(x)
        # print(x.shape)
        x = self.inception3a(x)
        # print(x.shape)
        x = self.inception3b(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.inception4a(x)
        # print(x.shape)
        if self.training:
            aux1 = self.aux1(x)  # <-- 这里

        x = self.inception4b(x)
        # print(x.shape)
        x = self.inception4c(x)
        # print(x.shape)
        x = self.inception4d(x)
        # print(x.shape)

        if self.training:
            aux2 = self.aux2(x)  # <-- 这里

        x = self.inception4e(x)
        # print(x.shape)
        x = self.inception5a(x)
        # print(x.shape)
        x = self.inception5b(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        main = self.fc(x)
        # print(main)
        if self.training:
            return main, aux1, aux2
        return main


if __name__ == "__main__":
    model = GoogleNet()
    x = torch.randn([1, 3, 32, 32])
    x, aux1, aux2 = model(x)
