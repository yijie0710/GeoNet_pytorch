from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
import DispNet
import FlowNet
import PoseNet
from sequence_folders import SequenceFolder
from loss_functions import *
from utils import *
from logger import *
import time
import csv
import os
from tensorboardX import SummaryWriter

n_iter = 0
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')


class GeoNetModel(object):
    def __init__(self, config, train_flow):
        global device
        self.config = config
        self.num_source = config['sequence_length'] - 1
        self.batch_size = config['batch_size']
        self.num_scales = torch.tensor(config['num_scales'])
        self.simi_alpha = torch.tensor(
            config['alpha_recon_image']).float().to(device)
        self.geometric_consistency_alpha = torch.tensor(
            config['geometric_consistency_alpha']).float().to(device)
        self.geometric_consistency_beta = torch.tensor(
            config['geometric_consistency_beta']).float().to(device)
        self.loss_weight_rigid_warp = torch.tensor(
            config['lambda_rw']).float().to(device)
        self.loss_weight_disparity_smooth = torch.tensor(
            config['lambda_ds']).float().to(device)
        self.loss_weight_full_warp = torch.tensor(
            config['lambda_fw']).float().to(device)
        self.loss_weigtht_full_smooth = torch.tensor(
            config['lambda_fs']).float().to(device)
        self.loss_weight_geometrical_consistency = torch.tensor(
            config['lambda_gc']).float().to(device)
        self.epochs = config['epochs']
        self.epoch_size = config['epoch_size']
        self.log = '{}/log_{}'.format(config['log_dir'], time.time())
        self.output_log_iter = self.config['save_log_iter']
        self.output_ckpt_iter = self.config['save_ckpt_iter']
        self.train_flow = train_flow
        ''' Data preparation
            TODO: transformation
        '''
        self.data_transform = None
        '''Nets preparation
        '''
        self.disp_net = DispNet.DispNet()
        self.pose_net = PoseNet.PoseNet(self.num_source)
        '''input channels:
            #src_views * (3 tgt_rgb + 3 src_rgb + 3 warp_rgb + 2 flow_xy +1 error )
        '''
        self.flow_net = FlowNet.FlowNet(12, self.config['flow_scale_factor'])
        if device.type == 'cuda':
            self.disp_net.cuda()
            self.pose_net.cuda()
            self.flow_net.cuda()

        if not self.train_flow and (not os.path.isfile(
                self.config['ckpt_path'])):
            self.disp_net.init_weight()
            self.pose_net.init_weight()
        else:
            ckpt = torch.load(self.config['ckpt_path'])
            self.disp_net.load_state_dict(ckpt['disp_net_state_dict'])
            self.pose_net.load_state_dict(ckpt['pose_net_state_dict'])
            if train_flow:
                if 'flow_net_state_dict' in ckpt:
                    self.flow_net.load_state_dict(ckpt['flow_net_state_dict'])
                else:
                    self.flow_net.init_weight()

        self.disp_net = torch.nn.DataParallel(self.disp_net)
        self.pose_net = torch.nn.DataParallel(self.pose_net)

        self.nets = {
            'disp': self.disp_net,
            'pose': self.pose_net,
            'flow': self.flow_net
        }

    def iter_data_preparation(self, sampled_batch):
        # sampled_batch: tgt_view, src_views, intrinsics
        # shape: batch,chnls h,w
        tgt_view = sampled_batch[0]
        # shape: batch,num_source,chnls,h,w
        src_views = sampled_batch[1]
        # shape: batch,3,3
        intrinsics = sampled_batch[2]

        # to device
        # shape: #batch,3,h,w
        self.tgt_view = tgt_view.to(device).float()
        self.src_views = src_views.to(device).float()
        self.intrinsics = intrinsics.to(device).float()
        # Assumme src_views is stack and the shapes is #batch,#3*#src_views,h,w
        # shape: #batch*#src_views,3,h,w
        self.src_views_concat = torch.cat([
            self.src_views[:, 3 * s:3 * (s + 1), :, :]
            for s in range(self.num_source)
        ],
                                          dim=0)

        #　shape:  #scale, #batch, #chnls, h,w
        self.tgt_view_pyramid = scale_pyramid(self.tgt_view, self.num_scales)
        #　shape:  #scale, #batch*#src_views, #chnls,h,w
        self.tgt_view_tile_pyramid = [
            self.tgt_view_pyramid[scale].repeat(self.num_source, 1, 1, 1)
            for scale in range(self.num_scales)
        ]

        #　shape:  # scale,#batch*#src_views, # chnls, h,w
        self.src_views_pyramid = scale_pyramid(self.src_views_concat,
                                               self.num_scales)

        # output multiple disparity prediction
        self.multi_scale_intrinsices = compute_multi_scale_intrinsics(
            self.intrinsics, self.num_scales)

    def build_dispnet(self):
        #       shape: batch,chnls,h,w
        self.dispnet_inputs = self.tgt_view

        # for multiple disparity prediction,
        # cat tgt_view and src_views along the batch dimension

        for s in range(self.num_source):
            self.dispnet_inputs = torch.cat(
                (self.dispnet_inputs,
                 self.src_views[:, 3 * s:3 * (s + 1), :, :]),
                dim=0)

        # shape: pyramid_scales, #batch+#batch*#src_views, h,w
        self.disparities = self.disp_net(self.dispnet_inputs)
        # TODO: spatial normalize the predict disparities

        # shape: pyramid_scales, bs, h,w
        self.depth = [1 / disp for disp in self.disparities]
        self.depth = [d.squeeze_(1) for d in self.depth]

    def build_posenet(self):
        self.posenet_inputs = torch.cat((self.tgt_view, self.src_views), dim=1)
        self.poses = self.pose_net(self.posenet_inputs)

    def build_rigid_warp_flow(self):
        # NOTE: this should be a python list,
        # since the sizes of different level of the pyramid are not same
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []

        for scale in range(self.num_scales):

            for src in range(self.num_source):
                fwd_rigid_flow = compute_rigid_flow(
                    self.poses[:, src, :],
                    self.depth[scale][:self.batch_size, :, :],
                    self.multi_scale_intrinsices[:, scale, :, :], False)
                bwd_rigid_flow = compute_rigid_flow(
                    self.poses[:, src, :],
                    self.depth[scale][self.batch_size * (
                        src + 1):self.batch_size * (src + 2), :, :],
                    self.multi_scale_intrinsices[:, scale, :, :], True)
                if not src:
                    fwd_rigid_flow_cat = fwd_rigid_flow
                    bwd_rigid_flow_cat = bwd_rigid_flow
                else:
                    fwd_rigid_flow_cat = torch.cat(
                        (fwd_rigid_flow_cat, fwd_rigid_flow), dim=0)
                    bwd_rigid_flow_cat = torch.cat(
                        (bwd_rigid_flow_cat, bwd_rigid_flow), dim=0)

            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_cat)
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_cat)

        self.fwd_rigid_warp_pyramid = [
            flow_warp(self.src_views_pyramid[scale],
                      self.fwd_rigid_flow_pyramid[scale])
            for scale in range(self.num_scales)
        ]

        self.bwd_rigid_warp_pyramid = [
            flow_warp(self.tgt_view_tile_pyramid[scale],
                      self.bwd_rigid_flow_pyramid[scale])
            for scale in range(self.num_scales)
        ]

        self.fwd_rigid_error_pyramid = [
            image_similarity(self.simi_alpha,
                             self.tgt_view_tile_pyramid[scale],
                             self.fwd_rigid_warp_pyramid[scale])
            for scale in range(self.num_scales)
        ]
        self.bwd_rigid_error_pyramid = [
            image_similarity(self.simi_alpha, self.src_views_pyramid[scale],
                             self.bwd_rigid_warp_pyramid[scale])
            for scale in range(self.num_scales)
        ]
        # from IPython import embed
        # from matplotlib import pyplot as plt
        # embed()

    def build_flownet(self):

        # output residual flow
        # TODO: non residual mode
        #   make input of the flowNet
        # cat along the color channels
        # shapes: #batch*#src_views, 3+3+3+2+1,h,w

        fwd_flownet_inputs = torch.cat(
            (self.tgt_view_tile_pyramid[0], self.src_views_pyramid[0],
             self.fwd_rigid_warp_pyramid[0], self.fwd_rigid_flow_pyramid[0],
             L2_norm(self.fwd_rigid_error_pyramid[0], dim=1)),
            dim=1)
        bwd_flownet_inputs = torch.cat(
            (self.src_views_pyramid[0], self.tgt_view_tile_pyramid[0],
             self.bwd_rigid_warp_pyramid[0], self.bwd_rigid_flow_pyramid[0],
             L2_norm(self.bwd_rigid_error_pyramid[0], dim=1)),
            dim=1)

        # shapes: # batch
        flownet_inputs = torch.cat((fwd_flownet_inputs, bwd_flownet_inputs),
                                   dim=0)

        # shape: (#batch*2, (3+3+3+2+1)*#src_views, h,w)
        self.resflow = self.flow_net(flownet_inputs)

    def build_full_warp_flow(self):
        # unnormalize the pyramid flow back to pixel metric
        resflow_scaling = []
        for s in range(self.num_scales):
            batch_size, _, h, w = self.resflow[s].shape
            # create a scale factor matrix for pointwise multiplication
            # NOTE: flow channels x,y
            scale_factor = torch.tensor([w, h]).view(1, 2, 1,
                                                     1).float().to(device)
            scale_factor = scale_factor.repeat(batch_size, 1, h, w)
            resflow_scaling.append(self.resflow[s] * scale_factor)

        self.resflow = resflow_scaling

        self.fwd_full_flow_pyramid = [
            self.resflow[s][:self.batch_size * self.num_source] +
            self.fwd_rigid_flow_pyramid[s] for s in range(self.num_scales)
        ]
        self.bwd_full_flow_pyramid = [
            self.resflow[s][self.batch_size * self.num_source:] +
            self.bwd_rigid_flow_pyramid[s] for s in range(self.num_scales)
        ]

        self.fwd_full_warp_pyramid = [
            flow_warp(self.src_views_pyramid[s], self.fwd_full_flow_pyramid[s])
            for s in range(self.num_scales)
        ]
        self.bwd_full_warp_pyramid = [
            flow_warp(self.tgt_view_tile_pyramid[s],
                      self.bwd_full_flow_pyramid[s])
            for s in range(self.num_scales)
        ]

        self.fwd_full_error_pyramid = [
            image_similarity(self.simi_alpha, self.fwd_full_warp_pyramid[s],
                             self.tgt_view_tile_pyramid[s])
            for s in range(self.num_scales)
        ]
        self.bwd_full_error_pyramid = [
            image_similarity(self.simi_alpha, self.bwd_full_warp_pyramid[s],
                             self.src_views_pyramid[s])
            for s in range(self.num_scales)
        ]

    def build_losses(self):
        # NOTE: geometrical consistency
        if self.train_flow:
            bwd2fwd_flow_pyramid = [
                flow_warp(self.bwd_full_flow_pyramid[s],
                          self.fwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]
            fwd2bwd_flow_pyramid = [
                flow_warp(self.fwd_full_flow_pyramid[s],
                          self.bwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]

            fwd_flow_diff_pyramid = [
                torch.abs(bwd2fwd_flow_pyramid[s] +
                          self.fwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]
            bwd_flow_diff_pyramid = [
                torch.abs(fwd2bwd_flow_pyramid[s] +
                          self.bwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]

            fwd_consist_bound_pyramid = [
                self.geometric_consistency_beta * self.fwd_full_flow_pyramid[s]
                * 2**s for s in range(self.num_scales)
            ]
            bwd_consist_bound_pyramid = [
                self.geometric_consistency_beta * self.bwd_full_flow_pyramid[s]
                * 2**s for s in range(self.num_scales)
            ]
            # stop gradient at maximum opeartions
            fwd_consist_bound_pyramid = [
                torch.max(s,
                          self.geometric_consistency_alpha).clone().detach()
                for s in fwd_consist_bound_pyramid
            ]

            bwd_consist_bound_pyramid = [
                torch.max(s,
                          self.geometric_consistency_alpha).clone().detach()
                for s in bwd_consist_bound_pyramid
            ]

            fwd_mask_pyramid = [(fwd_flow_diff_pyramid[s] * 2**s <
                                 fwd_consist_bound_pyramid[s]).float()
                                for s in range(self.num_scales)]
            bwd_mask_pyramid = [(bwd_flow_diff_pyramid[s] * 2**s <
                                 bwd_consist_bound_pyramid[s]).float()
                                for s in range(self.num_scales)]
        # from IPython import embed
        # from matplotlib import pyplot as plt
        # embed()
        # NOTE: loss
        self.loss_rigid_warp = 0
        self.loss_disp_smooth = 0
        if self.train_flow:
            self.loss_full_warp = 0
            self.loss_full_smooth = 0
            self.loss_geometric_consistency = 0

        for s in range(self.num_scales):

            self.loss_rigid_warp += self.loss_weight_rigid_warp *\
                self.num_source/2*(
                    torch.mean(self.fwd_rigid_error_pyramid[s]) +
                    torch.mean(self.bwd_rigid_error_pyramid[s]))

            self.loss_disp_smooth += self.loss_weight_disparity_smooth/2**s *\
                smooth_loss(self.disparities[s], torch.cat(
                    (self.tgt_view_pyramid[s], self.src_views_pyramid[s]), dim=0))

            if self.train_flow:
                self.loss_full_warp += self.loss_weight_full_warp * self.num_source / 2 * (
                    torch.sum(
                        torch.mean(fwd_full_error_pyramid[s], 1, True) *
                        fwd_mask_pyramid[s]) / torch.mean(fwd_mask_pyramid[s])
                    + torch.sum(
                        torch.mean(bwd_full_error_pyramid[s], 1, True) *
                        bwd_mask_pyramid[s]) / torch.mean(bwd_mask_pyramid[s]))

                self.loss_full_smooth += self.loss_weigtht_full_smooth/2**(s+1) *\
                    (flow_smooth_loss(
                        self.fwd_full_flow_pyramid[s], self.tgt_view_tile_pyramid[s]) +
                        flow_smooth_loss(self.bwd_full_flow_pyramid[s], self.src_views_pyramid[s]))

                self.loss_geometric_consistency += self.loss_weight_geometrical_consistency / 2 * (
                    torch.sum(
                        torch.mean(fwd_flow_diff_pyramid[s], 1, True) *
                        fwd_mask_pyramid[s]) / torch.mean(fwd_mask_pyramid[s])
                    + torch.sum(
                        torch.mean(bwd_flow_diff_pyramid[s], 1, True) *
                        bwd_mask_pyramid[s]) / torch.mean(bwd_mask_pyramid[s]))

        self.loss_total = self.loss_rigid_warp + self.loss_disp_smooth
        if self.train_flow:
            self.loss_total += self.loss_full_warp + \
                self.loss_full_smooth + self.loss_geometric_consistency

    def training_inside_epoch(self):
        global n_iter

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(precision=4)

        end = time.time()

        for i, sampled_batch in enumerate(self.train_loader):
            # for name, param in self.disp_net.named_parameters():
            #     if param.requires_grad:
            #         print(name, torch.mean(param.data))
            # for name, param in self.pose_net.named_parameters():
            #     if param.requires_grad:
            #         print(name, torch.mean(param.data))
            data_time.update(time.time() - end)
            self.iter_data_preparation(sampled_batch)

            self.build_dispnet()
            self.build_posenet()
            self.build_rigid_warp_flow()
            if self.train_flow:
                self.build_flownet()
                self.build_full_warp_flow()
            self.build_losses()

            self.optimizer.zero_grad()
            self.loss_total.backward()
            self.optimizer.step()

            #  log
            n_iter += 1
            losses.update(self.loss_total.item(), self.batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            #  write
            with open(self.log, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                if self.train_flow:
                    writer.writerow([
                        self.loss_total.item(),
                        self.loss_rigid_warp.item(),
                        self.loss_disp_smooth.item(),
                        self.loss_full_warp.item(),
                        self.loss_full_smooth.item(),
                        self.loss_geometric_consistency.item()
                    ])
                else:
                    writer.writerow([
                        self.loss_total.item(),
                        self.loss_rigid_warp.item(),
                        self.loss_disp_smooth.item()
                    ])
            self.logger.train_bar.update(i + 1)
            if i % self.output_log_iter == 0:
                self.logger.train_writer.write(
                    'Train: Time {} Data {} Loss {}'.format(
                        batch_time, data_time, losses))
            if i % self.output_ckpt_iter == 0:
                path = '{}/{}_{}'.format(
                    self.config['ckpt_dir'],
                    'flow' if self.train_flow else 'rigid', i)
                torch.save({
                    'iter': i,
                    'disp_net_state_dict': self.disp_net.state_dict(),
                    'pose_net_state_dict': self.pose_net.state_dict(),
                    'loss': self.loss_total
                }, path)
            if i >= self.epoch_size - 1:
                break

        return losses.avg[0]  # TODO: why not torch.mean(losses.avg) ?

    def train(self):
        global n_iter
        if not self.train_flow:
            self.pose_net.train()
            self.disp_net.train()

        self.train_set = SequenceFolder(
            self.config['data'],
            transform=self.data_transform,
            split='min_train',
            seed=self.config['seed'],
            img_height=self.config['img_height'],
            img_width=self.config['img_width'],
            sequence_length=self.config['sequence_length'])

        self.val_set = SequenceFolder(
            self.config['data'],
            transform=None,
            split='min_val',
            seed=self.config['seed'],
            img_height=self.config['img_height'],
            img_width=self.config['img_width'],
            sequence_length=self.config['sequence_length'])

        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            shuffle=True,
            drop_last=True,
            num_workers=self.config['data_workers'],
            batch_size=self.config['batch_size'],
            pin_memory=False)

        self.val_loader = torch.utils.data.DataLoader(
            self.val_set,
            shuffle=True,
            batch_size=1,
            drop_last=True,
            num_workers=self.config['data_workers'],
            pin_memory=False)

        optim_params = [{
            'params': v.parameters(),
            'lr': self.config['learning_rate']
        } for v in self.nets.values()]

        self.optimizer = torch.optim.Adam(
            optim_params,
            betas=(self.config['momentum'], self.config['beta']),
            weight_decay=self.config['weight_decay'])

        self.logger = TermLogger(
            n_epochs=self.config['epoch'],
            train_size=min(len(self.train_loader), self.config['epoch_size']),
            valid_size=len(self.val_loader))
        self.logger.epoch_bar.start()

        for epoch in range(self.epochs):
            self.logger.epoch_bar.update(epoch)
            self.logger.reset_train_bar()
            epoch_train_loss = self.training_inside_epoch()
            self.logger.train_writer.write(
                ' * Avg Loss : {:.3f}'.format(epoch_train_loss))

            self.logger.reset_valid_bar()
            epoch_val_loss = self.validate_inside_epoch_without_gt()
            self.logger.valid_writer.write(
                ' * Avg Loss : {:.3f}'.format(epoch_val_loss))

    def test(self):
        pass

    @torch.no_grad()
    def validate_inside_epoch_without_gt(self):
        global device

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(precision=4)

        end = time.time()

        self.disp_net.eval()
        self.pose_net.eval()
        if self.train_flow:
            self.flow_net.eval()
        
        for i, sampled_batch in enumerate(self.val_loader):
            data_time = time.time()-end
            start_time = time.time()
            self.iter_data_preparation(sampled_batch)
            self.build_dispnet()
            self.build_posenet()
            from IPython import embed
            embed()
            self.build_rigid_warp_flow()
            if self.train_flow:
                self.build_flownet()
                self.build_full_warp_flow()
            self.build_losses()

            losses.update(self.loss_total)
            batch_time.update(time.time()-end)
            data_time.update(data_time)
            end = time.time()

            with open(self.log, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                if self.train_flow:
                    writer.writerow([
                        self.loss_total.item(),
                        self.loss_rigid_warp.item(),
                        self.loss_disp_smooth.item(),
                        self.loss_full_warp.item(),
                        self.loss_full_smooth.item(),
                        self.loss_geometric_consistency.item()
                    ])
                else:
                    writer.writerow([
                        self.loss_total.item(),
                        self.loss_rigid_warp.item(),
                        self.loss_disp_smooth.item()
                    ])
            self.logger.val_writer.write(
                'Train: Time {} Data {} Loss {}'.format(
                    batch_time, data_time, losses))

        return losses.avg[0]

    @torch.no_grad()
    def validate_with_gt(self):
        pass
