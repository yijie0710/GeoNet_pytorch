# 1. create data loader for KITTI raw or KITTK odometry
# 2. dump examples
# 3. split data into train and val set
import argparse
import scipy.misc
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from path import Path
from imageio import imread,imsave
import os

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir", metavar='DIR',
                    help='path to original dataset')
parser.add_argument("--dataset-format", type=str, default='kitti', choices=["kitti", "cityscapes"])
parser.add_argument("--static-frames", default=None,
                    help="list of imgs to discard for being static, if not set will discard them based on speed \
                    (careful, on KITTI some frames have incorrect speed)")
parser.add_argument("--with-depth", action='store_true',
                    help="If available (e.g. with KITTI), will store depth ground truth along with images, for validation")
parser.add_argument("--with-pose", action='store_true',
                    help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation")
parser.add_argument("--no-train-gt", action='store_true',
                    help="If selected, will delete ground truth depth to save space")
parser.add_argument("--dump-root", type=str, default='dump', help="Where to dump the data")
parser.add_argument("--height", type=int, default=128, help="image height")
parser.add_argument("--width", type=int, default=416, help="image width")
parser.add_argument("--depth-size-ratio", type=int, default=1, help="will divide depth size by that ratio")
parser.add_argument("--num-threads", type=int, default=4, help="number of threads to use")

args = parser.parse_args()

def concat_image_sequence(img_paths):
    '''
     the order of image_paths:
        I_t-n, I_t-(n-1), ...,I_t-1, I_t, I_t+1, ..., I_t+(n-1),I_t+n 
        I_t is the target view while the others are the src_views
    '''
    imgs = []
    for i in img_paths:
        im = np.array(imread(i))
        imgs.append(im)
    # concat
    img_seq = np.concatenate(imgs,dim=1)
    return img_seq
    
def dump_example(n,dump_dir):
    # TODO: show progress

    example = data_loader.get_example_by_idx(n)
    if not example: 
        return

    img_seq = concat_image_sequence(example['img_seq_paths'])
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    img_path = '{}/{}.jpg'.format(dump_dir,example['file_name'])
    cam_path = '{}/{}.cam'.format(dump_dir, example['file_name'])
    
    # dump image sequences into jpg
    imsave(img_path,img_seq)

    # dump intrinsics into cam txt
    with open(cam_path,'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy))


# def dump_example(scene,dump_root):
#     scene_list = data_loader.collect_scenes(scene)
#     for scene_data in scene_list:
#         dump_dir = dump_root/scene_data['rel_path']
#         dump_dir.makedirs_p()
#         intrinsics = scene_data['intrinsics']

#         dump_cam_file = dump_dir/'cam.txt'

#         np.savetxt(dump_cam_file, intrinsics)
#         poses_file = dump_dir/'poses.txt'
#         poses = []

#         for sample in data_loader.get_scene_imgs(scene_data):
#             img, frame_nb = sample["img"], sample["id"]
#             dump_img_file = dump_dir/'{}.jpg'.format(frame_nb)
#             scipy.misc.imsave(dump_img_file, img)
#             if "pose" in sample.keys():
#                 poses.append(sample["pose"].tolist())
#             if "depth" in sample.keys():
#                 dump_depth_file = dump_dir/'{}.npy'.format(frame_nb)
#                 np.save(dump_depth_file, sample["depth"])
#         if len(poses) != 0:
#             np.savetxt(poses_file, np.array(poses).reshape(-1, 12), fmt='%.6e')

#         if len(dump_dir.files('*.jpg')) < 3:
#             dump_dir.rmtree()


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

    Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n)
                                      for n in range(data_loader.num_train))


    # split data into train and val
    # TODO: avoiding data snooping
    print('Generating train val lists')
    np.random.seed(8964)
    
    img_files = [ f for f in os.listdir(args.dump_root) if os.path.splitext(f)[-1]=='.jpg']
    canonic_prefixes = set([subdir.basename()[:-2] for subdir in subdirs])
    with open('{}/train.txt'.format(args.dump_root), 'w') as tf:
        with open('{}/val.txt'.format(args.dump_root), 'w') as vf:
            for f in img_files:
                if np.random.random()<0.1:
                    vf.write('{}/{}'.format(args.dump_root,
                                            os.path.splitext(f)[:-1]))
                else:
                    tf.write('{}/{}'.format(args.dump_root,
                                            os.path.splitext(f)[:-1]))


if __name__ == '__main__':
    main()
