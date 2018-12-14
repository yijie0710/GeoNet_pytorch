# modeified on https://github.com/yzcjtr/GeoNet
# and https://github.com/ClementPinard/SfmLearner-Pytorch

# 1. create data loader for KITTI raw or KITTK odometry
# 2. dump examples
# 3. split data into train and val set
import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path
from imageio import imread, imsave
import os
import time
end = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",   type=str, required=True, help="where the dataset is stored")
parser.add_argument("--dataset_name",  type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
parser.add_argument("--dump_root",     type=str, required=True, help="where to dump the data")
parser.add_argument("--seq_length",    type=int, required=True, help="length of each training sequence")
parser.add_argument("--img_height",    type=int, default=128,   help="image height")
parser.add_argument("--img_width",     type=int, default=416,   help="image width")
parser.add_argument("--num_threads",   type=int, default=16,     help="number of threads to use")
parser.add_argument("--remove_static", help="remove static frames from kitti raw data", action='store_true')

args = parser.parse_args()


def concat_image_sequence(imgs):
    '''
     the order of imagea:
        I_t-n, I_t-(n-1), ...,I_t-1, I_t, I_t+1, ..., I_t+(n-1),I_t+n 
        I_t is the target view while the others are the src_views
    '''
    img_seq = np.concatenate(imgs, axis=1)
    return img_seq


def dump_example(n, dump_dir):
    # TODO: show progress
    global end
    if n % 2000 == 0:
        print('Progress %d/%d time: %d s....' %
              (n, data_loader.num_train, time.time()-end))
        end = time.time()

    example = data_loader.get_example_by_idx(n)
    if not example:
        return

    img_seq = concat_image_sequence(example['image_seq'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    img_path = '{}/{}.jpg'.format(dump_dir, example['file_name'])
    cam_path = '{}/{}.cam'.format(dump_dir, example['file_name'])

    # dump image sequences into jpg
    imsave(img_path, img_seq)

    # dump intrinsics into cam txt
    with open(cam_path, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))

def main():
    dump_root = Path(args.dump_root)
    dump_root.mkdir_p()

    global data_loader

    if args.dataset_name == 'kitti_odom':
        from kitti_odom_loader import kitti_odom_loader
        data_loader = kitti_odom_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    if args.dataset_name == 'kitti_raw_eigen':
        from kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='eigen',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length,
                                       remove_static=args.remove_static)

    if args.dataset_name == 'kitti_raw_stereo':
        from kitti_raw_loader import kitti_raw_loader
        data_loader = kitti_raw_loader(args.dataset_dir,
                                       split='stereo',
                                       img_height=args.img_height,
                                       img_width=args.img_width,
                                       seq_length=args.seq_length,
                                       remove_static=args.remove_static)

    if args.dataset_name == 'cityscapes':
        from cityscapes_loader import cityscapes_loader
        data_loader = cityscapes_loader(args.dataset_dir,
                                        img_height=args.img_height,
                                        img_width=args.img_width,
                                        seq_length=args.seq_length)

    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n,args.dump_root)
                                      for n in range(data_loader.num_train))

    #split data into train and val
    #TODO: avoiding data snooping
    print('Generating train val lists')
    np.random.seed(8964)

    img_files = [f for f in os.listdir(
        args.dump_root) if os.path.splitext(f)[-1] == '.jpg']
    # canonic_prefixes = set([subdir.basename()[:-2] for subdir in subdirs])
    with open('{}/train.txt'.format(args.dump_root), 'w') as tf:
        with open('{}/val.txt'.format(args.dump_root), 'w') as vf:
            for f in img_files:
                assert len(os.path.splitext(f))==2
                if np.random.random() < 0.1:
                    vf.write('{}/{}\n'.format(args.dump_root,
                                            os.path.splitext(f)[0]))
                else:
                    tf.write('{}/{}\n'.format(args.dump_root,
                                            os.path.splitext(f)[0]))


if __name__ == '__main__':
    main()
