#!/usr/bin/python3
# Modified on https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/datasets/sequence_folders.py

import torch.utils.data as data
import numpy as np
from imageio import imread
# from path import Path
import random
import os
from joblib import Parallel, delayed


def make_sequence_views(img_path, sequence_length, width):
    views = np.array(imread(img_path)).astype(np.float64)
    w = views.shape[1]
    assert w == sequence_length*width
    demi_length = (sequence_length-1)//2
    tgt_view = np.array(
        views[:,width*demi_length:width*(demi_length+1),: ])

    # shape: (chnls, h, w)
    tgt_view = np.moveaxis(tgt_view,-1,0)

    src_ids = list(range(0, demi_length)) + \
        list(range(demi_length+1, sequence_length))
    src_views = [views[:, width*i:width*(i+1), :] for i in src_ids]
    # shape: (h, w, chnls)
    src_views = np.concatenate(src_views,axis=2)
    # shape: (chnls, h, w)
    src_views = np.moveaxis(src_views, -1, 0)

    return tgt_view, src_views


def make_instrinsics(cam_path):
    with open(cam_path, 'r') as f:
        intrinsics = np.array(f.readline().split()[0].split(
            ',')).astype(np.float32).reshape(3, 3)
    return intrinsics


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/split/1.jpg
        root/split/1.cam
        root/split/2.jpg
        root/split/2.cam
        ...

        An image sequence is stacked along the horizontal dimension of a image,
        where the order is t-n,t-(n-1),...t-1,t,t+1,...,t+(n-1),t+n.
        Therefore, the length of the image sequence is 2*n+1.
        I_t is the tgt_view while the others are the src_views.

        The intrinsics correspnonding to an image sequence X is recorded inside the X.cam,
        with 9 numbers of the 3*3 intrinsics.

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed, split, sequence_length, img_width, img_height, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = root

        self.split = split
        self.example_names = [name.split('\n')[0].split('/')[-1] for name in open(
            '{}/{}.txt'.format(self.root, self.split))]
        # self.data_folder = '{}/{}'.format(self.root, self.split)
        # if not os.path.exists(self.data_folder):
        #     raise ValueError(
        #         '{} set of {} does not exist'.format(self.split, self.root))
        if sequence_length % 2 == 0:
            raise ValueError(
                'sequence length should be odd while the input is {}'.format(sequence_length))
        self.sequence_length = sequence_length
        self.width = img_width
        self.img_height = img_height
        self.make_samples()

    def make_sample(self, i):
        if i % 200 == 0:
            print('progress: {}/{}'.format(i, len(self.imgs)))

        tgt_view, src_views = make_sequence_views(
            self.imgs[i], self.sequence_length, self.width)
        intrinsics = make_instrinsics(self.cams[i])

        '''
            the shapes of samples:
            tgt_view: (chnl, h, w)
            src_views, (self.sequence_length-1, chnls, h, w)
            intrinsics: (3, 3)
        '''
        sample = {'tgt_view': tgt_view,
                  'src_views': src_views, 'intrinsics': intrinsics}

        self.samples.append(sample)

    def make_samples(self):
        imgs = ['{}/{}.jpg'.format(self.root, name)
                for name in self.example_names]
        cams = ['{}/{}.cam'.format(self.root, name)
                for name in self.example_names]

        assert len(imgs) == len(cams)

        self.imgs = sorted(imgs)
        self.cams = sorted(cams)
        self.samples = []
        for i in range(len(self.imgs)):
            self.make_sample(i)
        random.shuffle(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return np.copy(sample['tgt_view']), np.copy(sample['src_views']), np.copy(sample['intrinsics'])

    def __len__(self):
        return len(self.samples)
