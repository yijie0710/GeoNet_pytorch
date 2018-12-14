#!/usr/bin/python3
# Modified on https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/datasets/sequence_folders.py

import torch.utils.data as data
import numpy as np
from imageio import imread
# from path import Path
import random
import os


def load_as_float(path):
    return imread(path).astype(np.float32)


def make_sequence_views(img_path):
    views = np.array(imread(img_path))
    views = np.moveaxis(views, -1, 0)
    w = views.shape[2]
    assert w == self.sequence_length*self.width
    demi_length = (self.sequence_length-1)//2
    tgt_view = np.array(
        views[:, :, self.width*demi_length:self.width*(demi_length+1)])
    src_ids = list(range(0, demi_length)+list(range(demi_length+1, self.sequence_length))
    # TODO: what's wrong?
    src_views=[views[:, :, self.width*i: self.width*(i+1)] for i in src_ids]
    src_views=np.array(src_views)
    return tgt_view, src_views

def make_instrinsics(cam_path):
    with open(cam_path, 'r') as f:
        intrinsics=f.readline().split()
    intrinsics=np.array(intrinsics.split(' ')).astype(float).reshape(3, 3)
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

    def __init__(self, root, seed=None, train=True, sequence_length=3, img_width, img_height, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root=root
        self.split='train' if train else 'val'
        self.data_folder='{}/{}'.format(self.root, split)
        if not os.path.exists(self.data_folder):
            raise ValueError(
                '{} set of {} does not exist'.format(split, self.root))
        if sequence_length % 2 == 0:
            raise ValueError(
                'sequence length should be odd while the input is {}'.format(sequence_length))
        self.sequence_length=sequence_length
        self.width=img_width
        self.img_height=img_height
        make_samples()


    def make_samples(self):
        all_files=os.listdir(self.data_folder)
        imgs=[]
        cams=[]
        for f in all_files:
            if os.path.splitext(f)[-1] == '.jpg':
                imgs.append(f)
            elif os.path.splitext(f)[-1] == '.cam':
                cams.append(f)

        assert len(imgs) == len(cams)

        imgs=sorted(imgs)
        cams=sorted(cams)
        self.samples=[]
        for i in range(len(imgs)):
            tgt_view, src_views=make_sequence_views(
                '{}/{}'.format(self.data_folder, imgs[i]))
            intrinsics=make_instrinsics(
                '{}/{}'.format(self.data_folder, cams[i]))

            '''
                the shapes of samples:
                tgt_view: (chnl, h, w)
                src_views, (self.sequence_length-1, chnls, h, w)
                intrinsics: (3, 3)
            '''
            sample={'tgt_view': tgt_view,
                'src_views': src_views, 'intrinsics': intrinsics}

            self.samples.append(sample)
        
        random.shuffle(self.samples)
        

    def __getitem__(self, index):
        sample=self.samples[index]
        return sample['tgt_view'], sample['src_views'], sample['intrinsics']

    def __len__(self):
        return len(self.samples)
