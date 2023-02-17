import torch
import torch.nn as nn

class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)

class resblock(nn.Module):
    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, 1),
                Conv_BN_LeakyReLU(ch//2, ch, 3, padding=1)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x

class YOLOv3Net(nn.Module):
    # 这里的YOLOv3Net可以接受不同尺寸的输入图片，但是要求输入图片是正方形，边长是32的倍数，而且是3通道的。
    # 它可以设置分类类别的数量。
    def __init__(self, classNumber):
        super().__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, padding=1),
            Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2),
            resblock(64, nblocks=1)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2),
            resblock(128, nblocks=2)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2),
            resblock(256, nblocks=8)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2),
            resblock(512, nblocks=8)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2),
            resblock(1024, nblocks=4)
        )
        # YOLOv3Net: a1
        self.layer_a1 = nn.Sequential(
            Conv_BN_LeakyReLU(1024, 512, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(512, 1024, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(1024, 512, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(512, 1024, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(1024, 512, 1, stride=1, padding=0)
        )
        # YOLOv3Net: a2
        self.layer_a2 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, stride=1, padding=1),
            nn.Conv2d(1024, (1 + 4 + classNumber) * 3, 1, stride=1, padding=0)
        )
        # YOLOv3Net: b1
        self.layer_b1 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 256, 1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        # YOLOv3Net: b2
        self.layer_b2 = nn.Sequential(
            Conv_BN_LeakyReLU(512 + 256, 256, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(256, 512, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(512, 256, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(256, 512, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(512, 256, 1, stride=1, padding=0)
        )
        # YOLOv3Net: b3
        self.layer_b3 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, stride=1, padding=1),
            nn.Conv2d(512, (1 + 4 + classNumber) * 3, 1, stride=1, padding=0)
        )
        # YOLOv3Net: c1
        self.layer_c1 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 128, 1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode="nearest")
        )
        # YOLOv3Net: c2
        self.layer_c2 = nn.Sequential(
            Conv_BN_LeakyReLU(256 + 128, 128, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(128, 256, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(256, 128, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(128, 256, 3, stride=1, padding=1),
            Conv_BN_LeakyReLU(256, 128, 1, stride=1, padding=0),
            Conv_BN_LeakyReLU(128, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, (1 + 4 + classNumber) * 3, 1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        f3 = x
        x = self.layer_4(x)
        f2 = x
        x = self.layer_5(x)
        f1 = x
        br1_1 = self.layer_a1(f1)
        br1_2o = self.layer_a2(br1_1)
        br2_1 = self.layer_b1(br1_1)
        br2_2 = torch.cat((br2_1, f2), dim=1)
        br2_3 = self.layer_b2(br2_2)
        br2_4o = self.layer_b3(br2_3)
        br3_1 = self.layer_c1(br2_3)
        br3_2 = torch.cat((br3_1, f3), dim=1)
        br3_3o = self.layer_c2(br3_2)

        return br1_2o, br2_4o, br3_3o
