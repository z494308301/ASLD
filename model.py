import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(dim, projection_size, hidden_size=64):

    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


class AzimuthHead(nn.Module):
    def __init__(self, in_channels=1024 * 5 * 5, mid_channels=256):
        super().__init__()

        # 🔥 关键：两层MLP降维，稳定、强特征、好收敛
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # 输出层（两个独立分支）
        self.fc_theta = nn.Linear(mid_channels, 1)  # 角度
        self.fc_kappa = nn.Linear(mid_channels, 1)  # 置信度

    def forward(self, x):
        # x: [B, 1024*5*5]
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 提取高质量低维特征

        # 输出
        theta_pred = self.fc_theta(x)  # 弧度
        kappa_pred = self.fc_kappa(x)
        conf = torch.sigmoid(kappa_pred)  # 置信度

        return theta_pred, conf

class PoseHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_1x1_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_1x1_conv(x)

class ClassHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_1x1_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_1x1_conv(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 上采样层（对应Encoder的down1-down4）
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, stride=1, kernel_size=7),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )  # 5×5→10×10
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )  # 10×10→22×22（与x4尺寸匹配）
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )  # 22×22→44×44（与x3匹配）
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )  # 44×44→88×88（与x2匹配）
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # 最终输出1通道（与输入一致）

    def forward(self, x_origin):  # 

        # x_origin: torch.Size([64, 2048, 5, 5])（cls_feature+pose_feature）
        x = self.up1(x_origin)  # [32,512,10,10]
        x = self.up2(x)  # [32,256,22,22]
        x = self.up3(x)  # [32,128,44,44]
        x = self.up4(x)  # [32,64,88,88]
        x = torch.sigmoid(self.final_conv(x))  # [32,1,88,88]
        return x

class DoubleConv(nn.Module):
    """(卷积 => 批归一化 => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样，包含一个最大池化层和一个 DoubleConv 模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样，包含一个反卷积层和一个 DoubleConv 模块"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 如果使用双线性插值进行上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 调整 x1 的大小以匹配 x2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpFeature(nn.Module):
    """上采样，包含一个反卷积层和一个 DoubleConv 模块"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 如果使用双线性插值进行上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x1 = self.up(x)
        # 调整 x1 的大小以匹配 x2

        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积层"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes = 10, m_classes = 2, bilinear=True):
        super(UNet, self).__init__()
        avgpool = 5

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.m_classes = m_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.cla = ClassHead(1024, 1024)
        self.pose = PoseHead(1024, 1024)

        self.MLP = MLP(dim=1024 * 5 * 5, projection_size = n_classes)
        self.avg = nn.AdaptiveMaxPool2d((avgpool, avgpool))

        self.pose_regressor = AzimuthHead(in_channels=1024 * 5 * 5)

        self.decoder = Decoder()


    def forward(self, x):  # input = torch.Size([32, 1, 88, 88])

        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # multiple layer features processed
        feature1 = torch.cat([self.avg(x1).repeat(1,2,1,1), self.avg(x2), self.avg(x3), self.avg(x4)], 1)\
            .mul(self.avg(x5).repeat(1, 2, 1, 1)) #  32, 1024, 5, 5

        # feature disentanglement
        cls_feature = self.cla(feature1)
        cls_project = cls_feature.view(x.size(0),-1)

        pose_feature = self.pose(feature1)
        pose_project = pose_feature.view(x.size(0),-1)

        # classification and regression
        predict_cls = self.MLP(cls_project) # 800, 10  torch.Size([32, 2]) torch.Size([32, 10])
        predict_ang, conf = self.pose_regressor(pose_project)


        x_origin = torch.cat((cls_feature, pose_feature), dim=1) #torch.Size([1, 1024, 5, 5]) torch.Size([1, 1024, 5, 5])

        reconstruction = self.decoder(x_origin)  # reconstruction = torch.Size([64, 2, 88, 88])


        return cls_project, pose_project, predict_cls, predict_ang, reconstruction, conf


