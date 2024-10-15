""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from USDM_utils.cbam import CBAM
from .usdmnet_parts import *


class USDM_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(USDM_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.shared_layers1 = DoubleConv(n_channels, 64, 128)
        self.shared_layers2 = Down(128, 256, 256)
        self.shared_layers3 = Down(256, 256, 256)

        self.up01 = Up(512, 256, 128, bilinear)
        self.up02 = Up(256, 128, 47, bilinear)
        self.DoubleConv_delta = DoubleConv(94, 64, 64)
        
        self.ConvDown_y01 = ConvDown(1, 2, 4)
        self.ConvDown_y02 = ConvDown(4, 8, 16)
        self.ConvDown_y03 = ConvDown(16, 32, 64)
        self.upscore1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # self.upscore2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.upscore3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_x1_1 = Up(512, 256, 128, bilinear)
        self.up_x1_2 = Up(256, 128, 47, bilinear)
        self.DoubleConv_x1_2 = DoubleConv(94, 64, 64)

        self.CBAMLayer1 = CBAM(n_channels_in=64,reduction_ratio=2,kernel_size=3)
        self.CBAMLayer2 = CBAM(n_channels_in=64,reduction_ratio=2,kernel_size=3)
        self.CBAMLayer3 = CBAM(n_channels_in=64,reduction_ratio=2,kernel_size=3)
        self.CBAMLayer4 = CBAM(n_channels_in=64,reduction_ratio=2,kernel_size=3)
        # self.CBAMLayer5 = CBAM(n_channels_in=128,reduction_ratio=4,kernel_size=3)
        # self.DoubleConv_all = DoubleConv(128,128)
        self.outc1 = OutConv(64, n_classes)
        self.outc2 = OutConv(64, n_classes)

        self.outc = OutConv(128, n_classes)

    def forward(self, x0, x1, y0):
        # y0 = torch.cat((y0_Mid,y0),dim=1)
        y_0 = self.ConvDown_y01(y0*0.2)
        y_0 = self.ConvDown_y02(y_0)
        y_0 = self.ConvDown_y03(y_0)
        y_0 = self.upscore1(y_0)
        y_0 = self.CBAMLayer1(y_0)

        # y_0 = self.ConvDown_y01(y0*0.2)
        # y_00 = self.ConvDown_y02(y_0)
        # y_000 = self.ConvDown_y03(y_00)
        # y_000 = self.upscore1(y_000) + self.upscore2(y_00)
        # y_000 = self.CBAMLayer1(y_000)
        
        x11 = self.shared_layers1(x1)
        x12 = self.shared_layers2(x11)
        x13 = self.shared_layers3(x12)

        x01 = self.shared_layers1(x0)
        x02 = self.shared_layers2(x01)
        x03 = self.shared_layers3(x02)

        delta_x = x1 - x0
        delta_x1 = x11 - x01
        delta_x2 = x12 - x02
        delta_x3 = x13 - x03
        delta_xOut = self.up01(delta_x3, delta_x2)
        delta_xOut = self.up02(delta_xOut, delta_x1)
        delta_xOut = torch.cat((delta_xOut, delta_x),dim=1)
        delta_xOut = self.DoubleConv_delta(delta_xOut)
        delta_xOut = self.CBAMLayer2(delta_xOut)
        y_delta = delta_xOut + y_0
        y_delta = self.CBAMLayer3(y_delta)

        x1_Out = self.up_x1_1(x13,x12)
        x1_Out = self.up_x1_2(x1_Out,x11)
        x1_Out = torch.cat((x1_Out, x1),dim=1)
        x1_Out = self.DoubleConv_x1_2(x1_Out)
        x1_Out = self.CBAMLayer4(x1_Out)

        x_y_delta = torch.cat((x1_Out, y_delta),dim=1)
        # x_y_delta = self.CBAMLayer5(x_y_delta)

        x_Outc1 = self.outc1(x1_Out)
        x_Outc2 = self.outc2(y_delta)
        logists = self.outc(x_y_delta)
        return  x_Outc1, x_Outc2, logists
    

# if __name__ == '__main__':
#     n_channels = 38
#     n_classes = 6
#     x0 = torch.zeros(1, 38, 512, 512)
#     x1 = torch.zeros(1, 38, 512, 512)
#     y0 = torch.zeros(1, 1, 512, 512)
#     net = USDM_Net(n_channels,n_classes)
#     outc1,outc2,out = net(x0,x1,y0)
