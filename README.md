# GeoNet_pytorch
This is an unofficial pytorch implementation of the paper:

GeoNet: Unsupervised Learning of Dense Depth, Optical Flow and Camera Pose (CVPR 2018)

Zhichao Yin and Jianping Shi

arxiv preprint: (https://arxiv.org/abs/1803.02276)

The official tensorflow implementation: https://github.com/yzcjtr/GeoNet

# Requirements
Build on:   
python 3.7
PyTorch 1.0 stable
CUDA 9.0
Ubuntu 16.04 / CentOS 7

# Training
This code follows the GeoNet authors stage-wise training as:
1. "train_rigid" mode: Train DispNet and PoseNe with rigid warp loss, smooth loss
2. "train_flow" mode: Fine tune/fix DispNet and PoseNet and train FlowNet with rigid warp loss, rigid smooth loss, fully warp loss, fully flow smooth loss and geometry consistency loss

# TODO
This repository is still work on progress and here are the todos:
- [ ] validation functions with ground truth
- [ ] test functions for depth, pose and optical flow
- [ ] evaluation 
- [ ] debugging and experiments


#Acknowledgements
This repository heavily reused codes from [SfmLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch)  by [Clement Pinard](https://github.com/ClementPinard). Many thanks to  [Clement Pinard](https://github.com/ClementPinard)'s work!
