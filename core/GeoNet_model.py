from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
import DispNet
import FlowNet
import PoseNet
from sequence_folders import SequenceFolder
from utils import *


class GeoNetModel(object):

    def __init__(self, mode, config, device):
        self.config = config
        self.mode = mode
        self.num_source = config['num_source']
        self.batch_size = config['batch_size']
        self.device = device
        self.num_scales = 4

        ''' Data preparation
            TODO: transformation
        '''
        data_transform = None
        self.dataset = SequenceFolder(
            config['data'],
            transform=data_transform,
            train=self.mode == 'train',
            seed=self.config['seed'],
            sequence_length=self.config['sequence_length']
        )

        '''Nets preparation
        '''
        self.disp_net = DispNet.DispNet()
        self.pose_net = PoseNet.PoseNet(config['num_source'])
        self.flow_net = FlowNet.FlowNet()
        self.nets = {'disp': self.disp_net,
                     'pose': self.pose_net,
                     'flow': self.flow_net}
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, shuffle=True,
            num_workers=config['num_workers'], batch_size=config['batch_size'])

        if mode == 'train':

            optim_params = [{'params': v.parameters(), 'lr': config['learning_rate']}
                            for v in self.nets.values()]
            optimizer = torch.optim.Adam(optim_params, betas=(
                config['momentum'], config['beta']))

    def train(self):
        for i, sample_batched in enumerate(self.data_loader):
            # shape: batch,chnls h,w
            tgt_view = sample_batched['tgt_img']
            # shape: batch,num_source,chnls,h,w
            src_views = sample_batched['src_imgs']
            # shape: batch,3,3
            intrinsics = sample_batched['intrinsics']

            # to device
            tgt_view = tgt_view.to(self.device)
            src_views = src_views.to(self.device)
            intrinsics = intrinsics.to(self.device)

            ＃　shape: #scale, #batch, # chnls, h,w
            tgt_view_pyramid = scale_pyramid(tgt_view,self.num_scales)
            ＃　shape:  #scale, #batch, #num_source,#chnls,h,w
            tgt_view_tile_pyramid = tgt_view_pyramid.repeat(self.num_source).permute(1,2,0,3,4,5)
            ＃　shape:  #scale,#batch, #num_source # chnls, h,w
            src_views_pyramid = scale_pyramid(src_views, self.num_scales)

            # output multiple disparity prediction
            multi_scale_intrinsices = compute_multi_scale_intrinsics(
                intrinsics, self.num_scales)

            #   make the input of the disparity prediction
            #       shape: batch,chnls,h,w
            dispnet_inputs = tgt_view.unsqueeze_(1)

            # for multiple disparity prediction,
            # we concat tgt_view and src_views along the batch dimension of inputs
            for s in range(self.num_source):
                dispnet_inputs = torch.cat(
                    (dispnet_inputs, src_views[:, s, :, :, :]), dim=1)
            dispnet_inputs.view(
                (self.batch_size*(1+self.num_source), tgt_view.shape[1], tgt_view.shape[2], tgt_view.shape[3]))

            # shape: pyramid_scales, bs, h,w
            disparities = self.disp_net(dispnet_inputs)
            # TODO: spatial normalize the predict disparities

            # shape: pyramid_scales, bs, h,w
            depth = [1/disp for disp in disparities]

            # output poses

            #    make the input of poseNet
            #   concat along the color dimension
            posenet_inputs = torch.cat((tgt_view, src_views), dim=1)
            poses = self.pose_net(posenet_inputs)

            # output rigid flow

            # NOTE: this should be a python list,
            # since the sizes of different level of the pyramid are not same
            fwd_rigid_flow_pyramid = []
            bwd_rigid_flow_pyramid = []

            for scale in range(self.num_scales):
                for src in range(self.num_source):
                    fwd_rigid_flow = compute_rigid_flow(
                        poses[:,src,:], depth[scale, :self.batch_size, :, :], multi_scale_intrinsices[:, scale, :, :], False)
                    bwd_rigid_flow = compute_rigid_flow(
                        poses[:,src,:], depth[scale, :self.batch_size, :, :], multi_scale_intrinsices[:, scale, :, :], True)
                    if not src:
                        fwd_rigid_flow_cat = fwd_rigid_flow
                        bwd_rigid_flow_cat = bwd_rigid_flow
                    else:
                        fwd_rigid_flow_cat = torch.cat((fwd_rigid_flow_cat,fwd_rigid_flow),dim=0)
                        bwd_rigid_flow_cat = torch.cat((bwd_rigid_flow_cat,bwd_rigid_flow),dim=0)
                fwd_rigid_flow_pyramid.append(fwd_rigid_flow_cat)
                bwd_rigid_flow_pyramid.append(bwd_rigid_flow_cat)
            
            fwd_rigid_warp_pyramid = [flow_warp() for scale in range(self.num_scales)]


                    # output residual flow

    def test(self):
        pass

    def inference(self):
        pass
