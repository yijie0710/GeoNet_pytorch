import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, zeros_


def resize_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


def upconv(in_chnls, out_chnls):
    return nn.Sequential(
        nn.ConvTranspose2d(in_chnls, out_chnls, kernel_size=3,
                           stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def downconv(in_chnls, out_chnls, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_chnls, out_chnls, kernel_size,
                  stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_chnls, out_chnls, kernel_size,
                  stride=1, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def conv(in_chnls, out_chnls):
    return nn.Sequential(
        nn.Conv2d(in_chnls, out_chnls, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

def get_flow(in_chnls):
    return nn.Conv2d(in_chnls,2,kernel_size=1,padding=0)


class FlowNet(nn.Module):
    def __init__(self,input_chnls,flow_scale_factor):
        super(FlowNet, self).__init__()
        # TODO: more inputs should be added
        #encode
        self.conv1 = downconv(input_chnls, 32, kernel_size=7)
        self.conv2 = downconv(32, 64, kernel_size=5)
        self.conv3 = downconv(64, 128, kernel_size=3)
        self.conv4 = downconv(128, 256, kernel_size=3)
        self.conv5 = downconv(256, 512, kernel_size=3)
        self.conv6 = downconv(512, 512, kernel_size=3)
        self.conv7 = downconv(512, 512, kernel_size=3)

        # decode
        self.upconv7 = upconv(512, 512)  # 128
        self.upconv6 = upconv(512, 512)  # 64
        self.upconv5 = upconv(512, 256)  # 32
        self.upconv4 = upconv(256, 128)  # 16
        self.upconv3 = upconv(128, 64)  # 8
        self.upconv2 = upconv(64, 32)  # 4
        self.upconv1 = upconv(32, 16)  # 2

        self.iconv7 = conv(512+512, 512)
        self.iconv6 = conv(512+512, 512)
        self.iconv5 = conv(256+256, 256)
        self.iconv4 = conv(128+128, 128)
        self.iconv3 = conv(64+64+2, 64)   # 64+64+2 in 1/4 out 1/4
        self.iconv2 = conv(32+32+2, 32)  #  32+32+2 in 1/2 out 1/2
        self.iconv1 = conv(16+2, 16)  #  16+2 in 1/1 out 1/1

        self.flow4 = get_flow(128)
        self.flow3 = get_flow(64)
        self.flow2 = get_flow(32)
        self.flow1 = get_flow(16)

        self.alpha = flow_scale_factor
        self.beta = 0
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_normal_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)
    
    def forward(self,x):
        #encode
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        # decode
        out_upconv7 = resize_like(self.upconv7(out_conv7), out_conv6)
        concat7 = torch.cat((out_upconv7, out_conv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = resize_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5),1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = resize_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = resize_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3),1)
        out_iconv4 = self.iconv4(concat4)
        out_flow4 = self.alpha*self.flow4(out_iconv4)+self.beta

        out_upconv3 = resize_like(self.upconv3(out_iconv4), out_conv2)
        out_upflow4 = resize_like(F.interpolate(
            out_flow4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, out_upflow4), 1)
        out_iconv3 = self.iconv3(concat3)
        out_flow3 = self.alpha*self.flow3(out_iconv3)+self.beta

        out_upconv2 = resize_like(self.upconv2(out_iconv3), out_conv1)
        out_upflow3 = resize_like(F.interpolate(
            out_flow3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, out_upflow3), 1)
        out_iconv2 = self.iconv2(concat2)
        out_flow2 = self.alpha*self.flow2(out_iconv2)+self.beta

        out_upconv1 = resize_like(self.upconv1(out_iconv2), x)
        out_upflow2 = resize_like(F.interpolate(
            out_flow2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, out_upflow2), 1)
        out_iconv1 = self.iconv1(concat1) 
        out_flow1 = self.alpha*self.flow1(out_iconv1)+self.beta

        return out_flow1, out_flow2, out_flow3, out_flow4

###################Test###################
