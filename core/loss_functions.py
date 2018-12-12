import torch
from utils import *

def image_similarity(alpha,x,y):
    return alpha*DSSIM(x,y)+(1-alpha)*torch.abs(x-y)

def smooth_loss(depth,image):
    gradient_depth_x = gradient_x(depth)  # (TODO)shape: bs,1,h,w
    gradient_depth_y = gradient_y(depth)

    gradient_img_x = gradient_x(image)  # (TODO)shape: bs,3,h,w
    gradient_img_y = gradient_y(image)

    exp_gradient_img_x = torch.exp(-torch.mean(torch.abs(gradient_img_x),1,True)) # (TODO)shape: bs,1,h,w
    exp_gradient_img_y = torch.exp(-torch.mean(torch.abs(gradient_img_y),1,True)) 

    smooth_x = gradient_depth_x*exp_gradient_img_x
    smooth_y = gradient_depth_y*exp_gradient_img_y

    return torch.mean(smooth_x+smooth_y)

def flow_smooth_loss(flow,img):
    # TODO two flows ?= rigid flow + object motion flow
    smoothness = 0
    for i in range(2):
        # TODO shape of flow: bs,channels(2),h,w
        smoothness += smooth_loss(flow[:, i, :, :].unsqueeze(-1), img)
    return smoothness/2

