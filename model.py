import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import SpatialGradient
from ssim import msssim
from torch import Tensor
import os
from PIL import Image
import cv2

from ssim import msssim

NUM_BANDS = 4

def conv3x3(in_channels, out_channels, stride=1,bias=True):
    return nn.Sequential(
        nn.ReplicationPad2d(1),   
        nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    )


def interpolate(inputs, size=None, scale_factor=None):
    return F.interpolate(inputs, size=size, scale_factor=scale_factor,
                         mode='bilinear', align_corners=True)
#边缘特征提取
class EdgeDetect(nn.Module):
    def __init__(self):
        super(EdgeDetect, self).__init__()
        self.spatial = SpatialGradient('diff')
        print(self.spatial)
        self.max_pool = nn.MaxPool2d(3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            conv3x3(4, 128, bias=True), nn.ReLU(True)
        )
    def forward(self, x: Tensor) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        y = self.max_pool(u)
        y = self.conv(y)
        return y

#通道注意力
class CALayer(nn.Module):
    def __init__(self,in_channels=128):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            conv3x3(in_channels, in_channels//8,bias=True),
            nn.ReLU(True),
            conv3x3(in_channels//8, in_channels,bias=True),
            nn.Sigmoid()          #输出为128×1*1(通道为128，长宽为1)
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)#.view(32,1,1,128)
        return x*y
#像素注意力
class PALayer(nn.Module):
    def __init__(self, in_channels=128):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                conv3x3(in_channels, in_channels//8,bias=True),
                nn.ReLU(True),
                conv3x3(in_channels//8, 1,bias=True),
                nn.Sigmoid()    #激活函数   输出为1*128*128
        )
    def forward(self, x):
        y = self.pa(x)#.view(32,128,128,1)
        return x*y

class CompoundLoss(nn.Module):
    def __init__(self, pretrained, alpha=0.8, normalize=True):
        super(CompoundLoss, self).__init__()
        self.pretrained = pretrained
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, prediction, target):
        
        return (F.mse_loss(prediction, target) +
                F.mse_loss(self.pretrained(prediction), self.pretrained(target)) +
                self.alpha * (1.0 - msssim(prediction, target,
                                           normalize=self.normalize)))
class CompoundLossDcstfn(nn.Module):
    def __init__(self, pretrained, alpha=0.8, normalize=True):
        super(CompoundLossDcstfn, self).__init__()
        self.pretrained = pretrained
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, prediction, target):
        
        return F.mse_loss(prediction, target)


#第一个编码器用于陆地卫星特征提取.
class FEncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128]
        super(FEncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True)
        )

#第二个编码器学习参考日期和预测日期之间的特征差异.
class REncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS * 3, 32, 64, 128]
        super(REncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3])
        )


class Decoder(nn.Sequential):
    def __init__(self):
        channels = [128, 64, 32, NUM_BANDS]
        super(Decoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            nn.Conv2d(channels[2], channels[3], 1)
        )



class Pretrained(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128, 128]
        super(Pretrained, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3], 2),
            nn.ReLU(True),
            conv3x3(channels[3], channels[4]),
            nn.ReLU(True)
        )


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.ed = EdgeDetect()
        self.encoder = FEncoder()
        self.residual = REncoder()
        self.calayer = CALayer()  # 非残差
        self.palayer = PALayer()
        self.decoder = Decoder()

    def forward(self, inputs):  
        inputs[0] = interpolate(inputs[0], scale_factor=1)  
        inputs[-1] = interpolate(inputs[-1], scale_factor=1)  
        prev_diff = self.residual(
            torch.cat((inputs[0], inputs[1], inputs[-1]), 1))  
        OEdge_diff = self.ed(inputs[1])  
        Ocf_diff = self.encoder(inputs[1])
        Ocf_diff = OEdge_diff + Ocf_diff + prev_diff
        Oca_diff = self.calayer(Ocf_diff)
        Opa_fuse = self.palayer(Oca_diff)
        if len(inputs) == 5:  
            inputs[2] = interpolate(inputs[2], scale_factor=1)  
            next_diff = self.residual(
                torch.cat((inputs[2], inputs[3], inputs[-1]), 1))  
            TEdge_diff = self.ed(inputs[3])  
            Tcf_diff = self.encoder(inputs[3])  
            Tcf_diff = TEdge_diff + Tcf_diff + next_diff
            Tca_fuse = self.calayer(Tcf_diff)
            Tpa_fuse = self.palayer(Tca_fuse)
            if self.training:
                prev_fusion = Opa_fuse
                next_fusion = Tpa_fuse
                return self.decoder(prev_fusion), self.decoder(next_fusion)
            else:
                one = inputs[0].new_tensor(1.0)  
                epsilon = inputs[0].new_tensor(1e-8)
                prev_dist = torch.abs(prev_diff) + epsilon
                next_dist = torch.abs(next_diff) + epsilon
                prev_mask = one.div(prev_dist).div(one.div(prev_dist) + one.div(next_dist))
                prev_mask = prev_mask.clamp_(0.0, 1.0)
                next_mask = one - prev_mask
                result = (prev_mask * Opa_fuse + next_mask * Tpa_fuse)
                result = self.decoder(result)
                return result
        else:
            return self.decoder(Opa_fuse)  # (inputs[1]) + prev_diff)   #输入L0进行编码特征提取    这个输入一组影像的实验结果

