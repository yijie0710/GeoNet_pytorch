import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, zeros_


def downconv(in_chnls,out_chnls, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_chnls, out_chnls, kernel_size=kernel_size,
                  stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


class PoseNet(nn.Module):

    def __init__(self, num_source):
        super(PoseNet, self).__init__()
        
        self.num_source = num_source

        self.conv1 = downconv(3*(1+num_source), 16, 7) #1/2
        self.conv2 = downconv(16, 32, 5) #1/4
        self.conv3 = downconv(32, 64, 3) #1/8
        self.conv4 = downconv(64, 128, 3) #1/16
        self.conv5 = downconv(128, 256, 3) #1/32
        self.conv6 = downconv(256, 256, 3) #1/64
        self.conv7 = downconv(256, 256, 3) # 1/128
        self.pred_poses = nn.Conv2d(256,6*self.num_source,kernel_size=1,padding=0) #1/128 shapes: bs,chnls,h,w

    def init_weight(self):
        pass
    
    def forward(self,x):
        
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        out_poses = self.pred_poses(out_conv7)
        out_avg_poses = torch.mean(out_poses,(2,3)) # shapes: bs,6*num_src,h,w-> bs,6*num_src
        out_avg_poses = 0.01* out_avg_poses.view(out_avg_poses.shape[0],self.num_source,6)
        return out_avg_poses

###################Test###################


