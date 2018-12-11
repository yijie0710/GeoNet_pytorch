import yaml
import time
import torch
import numpy as np
import DispNet 
import FlowNet
import PoseNet
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim
from loss_functions import *

n_iter = 0
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train(config, disp_net,pose_net,flow_net, train_loader, optimizer):
    global n_iter,device
    lambda_ds = config['lambda_ds']
    lambda_fw = config['lambda_fw']
    lambda_fs = config['lambda_fs']
    lambda_gc = config['lambda_gc']

    
    # set the net into the training mode
    disp_net.train()
    flow_net.train()
    pose_net.train()
    
    for i, (tgt_img, src_imgs, intrinsics) in enumerate(train_loader):
        #TODO: log time
        # intrinsics as matrix with intrinsics_inv or with a set of parameters?
        tgt_img = tgt_img.to(device)
        src_imgs = [s.to(device) for s in src_imgs]
        intrinsics = intrinsics.to(device)

        disparities = disp_net(tgt_img)
        depth = [1/disp for disp in disparities]
        pose  = pose_net(tgt_img,src_imgs)
        rigid_flow = 0
        res_flow = flow_net(tgt_img,src_imgs)
        full_flow = rigid_flow+res_flow
        # TODO:  to handle image pyramid
        # TODO: how to produce reconstruction results?

        # loss 
        loss_rw = 0
        loss_ds = smooth_loss(depth,tgt_img)
        loss_fw = 0
        loss_fs = flow_smooth_loss(full_flow,tgt_img)
        loss_gc = 0

        loss = loss_rw+lambda_ds*loss_ds+lambda_fw*loss_fw+lambda_fs*loss_fs+lambda_gc*loss_gc

        optimizer.zero_grad()
        loss.backword()
        optimizer.step()
        return loss
        


def main():
    import argparse
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--config',type=str)
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = yaml.load(f)
    print(config)

    train_set = None
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['data_workers'], pin_memory=True)
    
    disp_net = DispNet.DispNet()
    flow_net = FlowNet.FlowNet()
    pose_net = PoseNet.PoseNet(config['num_source'])
    optim_params = [
        {'params':disp_net.parameters(),'lr':config['learning_rate']},
        {'params':pose_net.parameters(),'lr':config['learning_rate']},
        {'params':flow_net.parameters(),'lr':config['learning_rate']}
    ]
    optimizer = torch.optim.Adam(optim_params,
        betas=(config['momentum'],config['beta']),
        weight_decay=config['weight_decay'])
    
    # training
    for epoch in range(config['epoch']):
        train_loss = train(config,disp_net,pose_net,flow_net,train_loader,optimizer)



if __name__ == "__main__":
    main()
