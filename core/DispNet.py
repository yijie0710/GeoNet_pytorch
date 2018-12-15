#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_,zeros_

def resize_like(input,ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

def upconv(in_chnls,out_chnls):
    return nn.Sequential(
        nn.ConvTranspose2d(in_chnls,out_chnls,kernel_size=3,stride=2,padding=1,output_padding=1),
        nn.ReLU(inplace=True)
    )

def downconv(in_chnls,out_chnls,kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_chnls,out_chnls,kernel_size,stride=2,padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_chnls,out_chnls,kernel_size,stride=1,padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )

def conv(in_chnls,out_chnls):
    return nn.Sequential(
        nn.Conv2d(in_chnls,out_chnls,kernel_size=3,padding=1),
        nn.ReLU(inplace=True)
    )

def get_disparity(in_chnls):
    return nn.Sequential(
        nn.Conv2d(in_chnls,1,kernel_size=3,padding=1),
        nn.Sigmoid()
    )

class DispNet(nn.Module):

    def __init__(self,alpha=10,beta=0.01):
        super(DispNet,self).__init__()
        # TODO: more inputs should be added
        # TODO: scaling the disparity
        self.alpha = alpha
        self.beta = beta
        #encode
        self.conv1 = downconv(3, 32, kernel_size=7)
        self.conv2 = downconv(32,64,kernel_size=5)
        self.conv3 = downconv(64,128,kernel_size=3)
        self.conv4 = downconv(128,256,kernel_size=3)
        self.conv5 = downconv(256,512,kernel_size=3)
        self.conv6 = downconv(512, 512, kernel_size=3)
        self.conv7 = downconv(512,512,kernel_size=3)

        # decode
        self.upconv7 = upconv(512,512) # 128
        self.upconv6 = upconv(512,512) # 64
        self.upconv5 = upconv(512,256) # 32
        self.upconv4 = upconv(256,128) # 16
        self.upconv3 = upconv(128,64) # 8
        self.upconv2 = upconv(64,32) # 4
        self.upconv1 = upconv(32,16) # 2

        self.iconv7 = conv(512+512,512)
        self.iconv6 = conv(512+512,512)
        self.iconv5 = conv(256+256,256)
        self.iconv4 = conv(128+128,128)
        self.iconv3 = conv(64+64+1,64)   # TODO: 64+64+2 in 1/4 out 1/4
        self.iconv2 = conv(32+32+1, 32)  # TODO: 32+32+2 in 1/2 out 1/2
        self.iconv1 = conv(16+1, 16)  # TODO: 16+2 in 1/1 out 1/1

        self.disp4 = get_disparity(128)
        self.disp3 = get_disparity(64)
        self.disp2 = get_disparity(32)
        self.disp1 = get_disparity(16)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
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
        out_upconv7 = resize_like(self.upconv7(out_conv7),out_conv6)
        concat7 = torch.cat((out_upconv7,out_conv6),1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = resize_like(self.upconv6(out_iconv7),out_conv5)
        concat6 = torch.cat((out_upconv6,out_conv5),1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = resize_like(self.upconv5(out_iconv6),out_conv4)
        concat5 = torch.cat((out_upconv5,out_conv4),1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4= resize_like(self.upconv4(out_iconv5),out_conv3)
        concat4 = torch.cat((out_upconv4,out_conv3),1)
        out_iconv4 = self.iconv4(concat4)
        out_disp4 = self.alpha*self.disp4(out_iconv4)+self.beta
        

        out_upconv3 = resize_like(self.upconv3(out_iconv4),out_conv2)
        out_updisp4 = resize_like(F.interpolate(
            out_disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, out_updisp4), 1)
        out_iconv3 = self.iconv3(concat3)
        out_disp3 = self.alpha*self.disp3(out_iconv3)+self.beta

        out_upconv2 = resize_like(self.upconv2(out_iconv3),out_conv1)
        out_updisp3 = resize_like(F.interpolate(
            out_disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, out_updisp3), 1)
        out_iconv2 = self.iconv2(concat2)
        out_disp2 = self.alpha*self.disp2(out_iconv2)+self.beta


        out_upconv1 = resize_like(self.upconv1(out_iconv2),x)
        out_updisp2 = resize_like(F.interpolate(out_disp2,scale_factor=2,mode='bilinear',align_corners=False),x)
        concat1 = torch.cat((out_upconv1, out_updisp2), 1)
        out_iconv1 = self.iconv1(concat1)
        out_disp1 = self.alpha*self.disp1(out_iconv1)+self.beta

        if self.training:
            return out_disp1,out_disp2,out_disp3,out_disp4
        else:
            return out_disp1
        
###################Test###################

        




        
