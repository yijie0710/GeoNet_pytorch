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
from loss_functions import *


class GeoNetModel(object):

    def __init__(self, mode, config, device):
        self.config = config
        self.mode = mode
        self.num_source = config['num_source']
        self.batch_size = config['batch_size']
        self.device = device
        self.num_scales = 4
        self.simi_alpha = config['alpha_image_similarity']
        self.geometric_consistency_alpha = config['geometric_consistency_alpha']
        self.geometric_consistency_beta = config['geometric_concsistency_beta']
        self.loss_weight_rigid_warp = config['lambda_rw']
        self.loss_weight_disparity_smooth = config['lambda_ds']
        self.loss_weight_full_warp = config['lambda_fw']
        self.loss_weigtht_full_smooth = config['lambda_fs']
        self.loss_weight_geometrical_consistency = config['lambda_gc']

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
            self.optimizer = torch.optim.Adam(optim_params, betas=(
                config['momentum'], config['beta']),
                weight_decay=config['weight_decay'])

    def train(self):
        for i, sample_batched in enumerate(self.data_loader):
            # shape: batch,chnls h,w
            tgt_view = sample_batched['tgt_img']
            # shape: batch,num_source,chnls,h,w
            src_views = sample_batched['src_imgs']
            # shape: batch,3,3
            intrinsics = sample_batched['intrinsics']

            # to device
            # shape: #batch,3,h,w
            tgt_view = tgt_view.to(self.device)
            src_views = src_views.to(self.device)
            intrinsics = intrinsics.to(self.device)
            # Assumme src_views is stack and the shapes is #batch,#3*#src_views,h,w
            # shape: #batch*#src_views,3,h,w
            src_views_concat = torch.cat([src_views[:, 3*s:3*(s+1), :, :]
                                          for s in range(self.num_source)], dim=0)

            #　shape:  #scale, #batch, #chnls, h,w
            tgt_view_pyramid = scale_pyramid(tgt_view, self.num_scales)
            #　shape:  #scale, #batch*#src_views, #chnls,h,w
            tgt_view_tile_pyramid = [tgt_view_pyramid[scale].repeat(self.num_source, 1, 1, 1)
                                     for scale in range(self.num_scales)]

            #　shape:  # scale,#batch*#src_views, # chnls, h,w
            src_views_pyramid = scale_pyramid(src_views_concat,
                                              self.num_scales)

            # output multiple disparity prediction
            multi_scale_intrinsices = compute_multi_scale_intrinsics(
                intrinsics, self.num_scales)

            #   make the input of the disparity prediction
            #       shape: batch,chnls,h,w
            dispnet_inputs = tgt_view

            # for multiple disparity prediction,
            # cat tgt_view and src_views along the batch dimension
            for s in range(self.num_source):
                dispnet_inputs = torch.cat(
                    (dispnet_inputs, src_views[:, 3*s:s*(s+1), :, :]), dim=0)

            # shape: pyramid_scales, #batch+#batch*#src_views, h,w
            disparities = self.disp_net(dispnet_inputs)
            # TODO: spatial normalize the predict disparities

            # shape: pyramid_scales, bs, h,w
            depth = [1/disp for disp in disparities]

            # output poses

            #    make the input of poseNet
            #   cat along the color dimension
            posenet_inputs = torch.cat((tgt_view, src_views), dim=1)
            poses = self.pose_net(posenet_inputs)

            # output rigid flow

            # NOTE: this should be a python list,
            # since the sizes of different level of the pyramid are not same
            fwd_rigid_flow_pyramid = []
            bwd_rigid_flow_pyramid = []

            for scale in range(self.num_scales):
                for src in range(self.num_source):
                    fwd_rigid_flow = compute_rigid_flow(poses[:, src, :], depth[scale, :self.batch_size, :, :],
                                                        multi_scale_intrinsices[:, scale, :, :], False)
                    bwd_rigid_flow = compute_rigid_flow(poses[:, src, :],
                                                        depth[scale, self.batch_size*(
                                                            src+1):src:self.batch_size*(src+2), :, :],
                                                        multi_scale_intrinsices[:, scale, :, :], True)
                    if not src:
                        fwd_rigid_flow_cat = fwd_rigid_flow
                        bwd_rigid_flow_cat = bwd_rigid_flow
                    else:
                        fwd_rigid_flow_cat = torch.cat(
                            (fwd_rigid_flow_cat, fwd_rigid_flow), dim=0)
                        bwd_rigid_flow_cat = torch.cat(
                            (bwd_rigid_flow_cat, bwd_rigid_flow), dim=0)
                fwd_rigid_flow_pyramid.append(fwd_rigid_flow_cat)
                bwd_rigid_flow_pyramid.append(bwd_rigid_flow_cat)

            fwd_rigid_warp_pyramid = [
                flow_warp(src_views_pyramid[scale], fwd_rigid_flow_pyramid[s])
                for scale in range(self.num_scales)]
            bwd_rigid_warp_pyramid = [
                flow_warp(tgt_view_tile_pyramid[s], bwd_rigid_flow_pyramid[s])
                for scale in range(self.num_scales)]

            fwd_rigid_error_pyramid = [image_similarity(self.simi_alpha, tgt_view_tile_pyramid[s], fwd_rigid_warp_pyramid[s])
                                       for scale in range(self.num_scales)]
            bwd_rigid_error_pyramid = [image_similarity(self.simi_alpha, src_views_pyramid[s], bwd_rigid_warp_pyramid[s])
                                       for scale in range(self.num_scales)]

            # output residual flow
            # TODO: non residual mode
            #   make input of the flowNet
            # cat along the color channels
            # shapes: #batch*#src_views, 3+3+3+2+1,h,w
            fwd_flownet_inputs = torch.cat(
                (tgt_view_tile_pyramid[0], src_views_pyramid[0],
                 fwd_rigid_warp_pyramid[0], fwd_rigid_flow_pyramid[0],
                 L2_norm(fwd_rigid_error_pyramid[0], dim=1)), dim=1)
            bwd_flownet_inputs = torch.cat(
                (src_views_pyramid[0], tgt_view_tile_pyramid[0],
                 bwd_rigid_warp_pyramid[0], bwd_rigid_flow_pyramid[0],
                 L2_norm(bwd_rigid_error_pyramid[0], dim=1)), dim=1)

            # shapes: # batch
            flownet_inputs = torch.cat((fwd_flownet_inputs,
                                        bwd_flownet_inputs), dim=0)

            # shape: #batch, #src, 2,h,w
            resflow = self.flow_net(flownet_inputs)

            # unnormalize the pyramid flow back to pixel metric
            for s in range(self.num_scales):
                batch_size, _, h, w = resflow[s].shape
                # create a scale factor matrix for pointwise multiplication
                # NOTE: flow channels x,y
                scale_factor = torch.tensor([w, h]).type(
                    torch.FloatTensor).view(1, 2, 1, 1)
                scale_factor = scale_factor.repeat(batch_size, 1, h, w)
                resflow[s] = resflow[s]*scale_factor

            fwd_full_flow_pyramid = [resflow[s][:self.batch_size*self.num_source]+fwd_rigid_flow_pyramid[s]
                                     for s in range(self.num_scales)]
            bwd_full_flow_pyramid = [resflow[s][self.batch_size*self.num_source:]+bwd_rigid_flow_pyramid[s]
                                     for s in range(self.num_scales)]

            fwd_full_warp_pyramid = [flow_warp(src_views_pyramid[s], fwd_full_flow_pyramid[s])
                                     for s in range(self.num_scales)]
            bwd_full_warp_pyramid = [flow_warp(tgt_view_tile_pyramid[s], bwd_full_flow_pyramid[s])
                                     for s in range(self.num_scales)]

            fwd_full_error_pyramid = [image_similarity(fwd_full_warp_pyramid[s], tgt_view_pyramid[s])
                                      for s in range(self.num_scales)]
            bwd_full_error_pyramid = [image_similarity(bwd_full_warp_pyramid[s], src_views_pyramid[s])
                                      for s in range(self.num_scales)]

            # NOTE: geometrical consistency
            bwd2fwd_flow_pyramid = [flow_warp(bwd_full_flow_pyramid, fwd_full_flow_pyramid)
                                    for s in range(self.num_scales)]
            fwd2bwd_flow_pyramid = [flow_warp(fwd_full_flow_pyramid, bwd_full_flow_pyramid)
                                    for s in range(self.num_scales)]

            fwd_flow_diff_pyramid = [torch.abs(bwd2fwd_flow_pyramid[s]+fwd_full_flow_pyramid[s])
                                     for s in range(self.num_scales)]
            bwd_flow_diff_pyramid = [torch.abs(fwd2bwd_flow_pyramid[s]+bwd_full_flow_pyramid[s])
                                     for s in range(self.num_scales)]

            fwd_consist_bound_pyramid = [self.geometric_consistency_beta*fwd_full_flow_pyramid[s]*2**s
                                         for s in range(self.num_scales)]
            bwd_consist_bound_pyramid = [self.geometric_consistency_beta*bwd_full_flow_pyramid[s]*2**s
                                         for s in range(self.num_scales)]
            # stop gradient at maximum opeartions
            fwd_consist_bound_pyramid = [torch.max(s, self.geometric_consistency_alpha).clone().detach()
                                         for s in fwd_consist_bound_pyramid]
            bwd_consist_bound_pyramid = [torch.max(s, self.geometric_consistency_alpha).clone().detach()
                                         for s in bwd_consist_bound_pyramid]

            fwd_mask = [torch.less(fwd_flow_diff_pyramid[s]*2**s, fwd_consist_bound_pyramid[s]).type(torch.FloatTensor)
                        for s in range(self.num_scales)]
            bwd_mask = [torch.less(bwd_flow_diff_pyramid[s]*2**s, bwd_consist_bound_pyramid[s]).type(torch.FloatTensor)
                        for s in range(self.num_scales)]

            for scale in range(self.num_scales):
                for src in range(self.num_source):
                    fwd_rigid_flow = compute_rigid_flow(poses[:, src, :], depth[scale, :self.batch_size, :, :],
                                                        multi_scale_intrinsices[:, scale, :, :], False)
                    bwd_rigid_flow = compute_rigid_flow(poses[:, src, :],
                                                        depth[scale, self.batch_size*(
                                                            src+1):src:self.batch_size*(src+2), :, :],
                                                        multi_scale_intrinsices[:, scale, :, :], True)
                    if not src:
                        fwd_rigid_flow_cat = fwd_rigid_flow
                        bwd_rigid_flow_cat = bwd_rigid_flow
                    else:
                        fwd_rigid_flow_cat = torch.cat(
                            (fwd_rigid_flow_cat, fwd_rigid_flow), dim=0)
                        bwd_rigid_flow_cat = torch.cat(
                            (bwd_rigid_flow_cat, bwd_rigid_flow), dim=0)
                fwd_rigid_flow_pyramid.append(fwd_rigid_flow_cat)
                bwd_rigid_flow_pyramid.append(bwd_rigid_flow_cat)

            fwd_rigid_warp_pyramid = [
                flow_warp(src_views_pyramid[scale], fwd_rigid_flow_pyramid[s])
                for scale in range(self.num_scales)]
            bwd_rigid_warp_pyramid = [
                flow_warp(tgt_view_tile_pyramid[s], bwd_rigid_flow_pyramid[s])
                for scale in range(self.num_scales)]

            fwd_rigid_error_pyramid = [image_similarity(self.simi_alpha, tgt_view_tile_pyramid[s], fwd_rigid_warp_pyramid[s])
                                       for scale in range(self.num_scales)]
            bwd_rigid_error_pyramid = [image_similarity(self.simi_alpha, src_views_pyramid[s], bwd_rigid_warp_pyramid[s])
                                       for scale in range(self.num_scales)]

            # output residual flow
            # TODO: non residual mode
            #   make input of the flowNet
            # cat along the color channels
            # shapes: #batch*#src_views, 3+3+3+2+1,h,w
            fwd_flownet_inputs = torch.cat(
                (tgt_view_tile_pyramid[0], src_views_pyramid[0],
                 fwd_rigid_warp_pyramid[0], fwd_rigid_flow_pyramid[0],
                 L2_norm(fwd_rigid_error_pyramid[0], dim=1)), dim=1)
            bwd_flownet_inputs = torch.cat(
                (src_views_pyramid[0], tgt_view_tile_pyramid[0],
                 bwd_rigid_warp_pyramid[0], bwd_rigid_flow_pyramid[0],
                 L2_norm(bwd_rigid_error_pyramid[0], dim=1)), dim=1)

            # shapes: # batch
            flownet_inputs = torch.cat((fwd_flownet_inputs,
                                        bwd_flownet_inputs), dim=0)

            # shape: #batch, #src, 2,h,w
            resflow = self.flow_net(flownet_inputs)

            # unnormalize the pyramid flow back to pixel metric
            for s in range(self.num_scales):
                batch_size, _, h, w = resflow[s].shape
                # create a scale factor matrix for pointwise multiplication
                # NOTE: flow channels x,y
                scale_factor = torch.tensor([w, h]).type(
                    torch.FloatTensor).view(1, 2, 1, 1)
                scale_factor = scale_factor.repeat(batch_size, 1, h, w)
                resflow[s] = resflow[s]*scale_factor

            fwd_full_flow_pyramid = [resflow[s][:self.batch_size*self.num_source]+fwd_rigid_flow_pyramid[s]
                                     for s in range(self.num_scales)]
            bwd_full_flow_pyramid = [resflow[s][self.batch_size*self.num_source:]+bwd_rigid_flow_pyramid[s]
                                     for s in range(self.num_scales)]

            fwd_full_warp_pyramid = [flow_warp(src_views_pyramid[s], fwd_full_flow_pyramid[s])
                                     for s in range(self.num_scales)]
            bwd_full_warp_pyramid = [flow_warp(tgt_view_tile_pyramid[s], bwd_full_flow_pyramid[s])
                                     for s in range(self.num_scales)]

            fwd_full_error_pyramid = [image_similarity(fwd_full_warp_pyramid[s], tgt_view_pyramid[s])
                                      for s in range(self.num_scales)]
            bwd_full_error_pyramid = [image_similarity(bwd_full_warp_pyramid[s], src_views_pyramid[s])
                                      for s in range(self.num_scales)]

            # NOTE: geometrical consistency
            bwd2fwd_flow_pyramid = [flow_warp(bwd_full_flow_pyramid, fwd_full_flow_pyramid)
                                    for s in range(self.num_scales)]
            fwd2bwd_flow_pyramid = [flow_warp(fwd_full_flow_pyramid, bwd_full_flow_pyramid)
                                    for s in range(self.num_scales)]

            fwd_flow_diff_pyramid = [torch.abs(bwd2fwd_flow_pyramid[s]+fwd_full_flow_pyramid[s])
                                     for s in range(self.num_scales)]
            bwd_flow_diff_pyramid = [torch.abs(fwd2bwd_flow_pyramid[s]+bwd_full_flow_pyramid[s])
                                     for s in range(self.num_scales)]

            fwd_consist_bound_pyramid = [self.geometric_consistency_beta*fwd_full_flow_pyramid[s]*2**s
                                         for s in range(self.num_scales)]
            bwd_consist_bound_pyramid = [self.geometric_consistency_beta*bwd_full_flow_pyramid[s]*2**s
                                         for s in range(self.num_scales)]
            # stop gradient at maximum opeartions
            fwd_consist_bound_pyramid = [torch.max(s, self.geometric_consistency_alpha).clone().detach()
                                         for s in fwd_consist_bound_pyramid]
            bwd_consist_bound_pyramid = [torch.max(s, self.geometric_consistency_alpha).clone().detach()
                                         for s in bwd_consist_bound_pyramid]

            fwd_mask_pyramid = [torch.less(fwd_flow_diff_pyramid[s]*2**s, fwd_consist_bound_pyramid[s]).type(torch.FloatTensor)
                                for s in range(self.num_scales)]
            bwd_mask_pyramid = [torch.less(bwd_flow_diff_pyramid[s]*2**s, bwd_consist_bound_pyramid[s]).type(torch.FloatTensor)
                                for s in range(self.num_scales)]

            # NOTE: loss
            loss_rigid_warp = 0
            loss_disp_smooth = 0
            loss_full_warp = 0
            loss_full_smooth = 0
            loss_geometric_consistency = 0

            for scale in range(self.num_scales):
                loss_rigid_warp += self.loss_weight_rigid_warp *\
                    self.num_source/2*(
                        torch.mean(fwd_rigid_error_pyramid[s]) +
                        torch.mean(bwd_rigid_error_pyramid[s]))

                loss_disp_smooth += self.loss_weight_disparity_smooth/2**s *\
                    smooth_loss(disparities[s], torch.cat(
                        (tgt_view_pyramid[s], src_views_pyramid[s]), dim=0))

                loss_full_warp += self.loss_weight_full_warp*self.num_source/2 * \
                    (torch.mean(
                        fwd_full_error_pyramid[s])+torch.mean(bwd_full_error_pyramid[s]))

                loss_full_smooth += self.loss_weigtht_full_smooth/2**(s+1) *\
                    (flow_smooth_loss(
                        fwd_full_flow_pyramid[s], tgt_view_tile_pyramid[s]) +
                        flow_smooth_loss(bwd_full_flow_pyramid[s], src_views_pyramid[s]))

                loss_geometric_consistency += self.loss_weight_geometrical_consistency/2*(
                    +torch.sum(torch.mean(fwd_flow_diff_pyramid[s], 1, True)*fwd_mask_pyramid[s])
                    / torch.mean(fwd_mask_pyramid[s])
                    + torch.sum(torch.mean(bwd_flow_diff_pyramid[s], 1, True)*bwd_mask_pyramid[s])
                    / torch.mean(bwd_mask_pyramid[s]))

            loss_total = loss_rigid_warp+loss_disp_smooth+loss_full_warp+loss_full_smooth+loss_geometric_consistency
            
            self.optimizer.zero_grad()
            loss_total.backward()
            loss_total.step()

    def test(self):
        pass

    def inference(self):
        pass
