from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim
import DispNet
import FlowNet
import PoseNet
from sequence_folders import SequenceFolder


class GeoNetModel(object):

    def __init__(self, mode, config, device):
        self.config = config
        self.mode = mode

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

            optim_params = [['params': v.parameters(), 'lr':config['learning_rate']]
                            for v in self.nets.values()]
            optimizer = torch.optim.Adam(optim_params, betas=(
                config['momentum'], config['beta']))

    def train():
        for i, (tgt_view, src_views, intrinsics) in enumerate(self.data_loader):

    def test():
        pass

    def inference():
        pass
