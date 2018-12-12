import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math


def resize_2d(img, size):
    # Support resizin on GPU
    return (F.adaptive_avg_pool2d(Variable(img, volatile=True), size)).data


def scale_pyramid(img, num_scales):
    # shape of img: batch_size,channels,h,w
    if img is None:
        return None
    else:
        scaled_imgs = [img]
        # TODO: Assume the shape of image is [#channels, #rows, #cols ]
        h, w = img.shape[-2:]
        for i in range(num_scales-1):
            ratio = 2**(i+1)
            nh = int(h/ratio)
            nw = int(w/ratio)
            scaled_img = resize_2d(img, (nh, nw))
            scaled_imgs.append(scaled_img)
    # shape: #scales, # batch, #chnl, h, w
    return scaled_imgs


def L2_norm(self, x, dim, keep_dims=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset,
                         dim=dim, keep_dims=keep_dims)
    return l2_norm


def DSSIM(x, y):
    ''' Official implementation
    def SSIM(self, x, y):
        C1 = 0.01 ** 2 # why not use L=255
        C2 = 0.03 ** 2 # why not use L=255

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        # if this implementatin equvalent to the SSIM paper?
        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2 
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)
    '''
    # TODO: padding depend on the size of the input image sequences

    avepooling2d = torch.nn.AvgPool2d(3, stride=1, padding=[1, 1])
    mu_x = avepooling2d(x)
    mu_y = avepooling2d(y)
    sigma_x = avepooling2d((x-mu_x)**2)
    sigma_y = avepooling2d((y-mu_y)**2)
    sigma_xy = avepooling2d((x-mu_x)*(y-mu_y))
    k1_square = 0.01**2
    k2_square = 0.03**2
    L_square = 255**2
    SSIM_n = (2*mu_x*mu_y+k1_square*L_square)*(2*sigma_xy+k2_square*L_square)
    SSIM_d = (mu_x**2+mu_y**2+k1_square*L_square) * \
        (sigma_x+sigma_y+k2_square*L_square)
    SSIM = SSIM_n/SSIM_d
    return torch.clamp((1-SSIM)/2, 0, 1)


def gradient_x(img):
    return img[:, :, :, :-1]-img[:, :, :, 1:]


def gradient_y(img):
    return img[:, :, :-1, :]-img[:, :, 1:, :]


def residual_flow(intrinsics, T, Depth, pt):
    # BETTER to use KTDK'pt-pt as matrix multiple or by formulation as equations?
    pc = torch.tensor([(pt[0]-intrinsics.cx)*Depth/intrinsics.fx,
                       (pt[1]-intrinsics.cy)*Depth/intrinsics.fy,
                       Depth,
                       1],
                      requires_grad=True)
    pc_n = torch.matmul(T, pc)
    pt_n = torch.tensor([intrinsics.fx*pc_n[0]/pc_n[2]+intrinsics.cx,
                         intrinsics.fy*pc_n[1]/pc_n[2]+intrinsics.cy],
                        requires_grad=True)
    return pt_n-pt


def compute_multi_scale_intrinsics(intrinsics, num_scales):
    batch_size = intrinsics.shape[0]
    multi_scale_intrinsices = []
    for s in range(num_scales):
        fx = intrinsics[:, 0, 0]/(2**s)
        fy = intrinsics[:, 1, 1]/(2**s)
        cx = intrinsics[:, 0, 2]/(2**s)
        cy = intrinsics[:, 1, 2]/(2**s)
        r1 = torch.stack([fx, 0, cx], dim=1)  # shape: batch_size,3
        r2 = torch.stack([0, fy, cy], dim=1)  # shape: batch_size,3
        # shape: batch_size,3
        r3 = torch.tensor([0., 0., 1.]).view(1, 3).repeat(
            batch_size, 1)
        # concat along the spatial row dimension
        scale_instrinsics = torch.stack([r1, r2, r3], dim=1)
        multi_scale_intrinsices.append(
            scale_instrinsics)  # shape: num_scale,bs,3,3
    multi_scale_intrinsices = torch.stack(multi_scale_intrinsices, dim=1)
    return multi_scale_intrinsices


def euler2mat(z, y, x):
    # TODO: eular2mat
    '''
    input shapes of z,y,x all are: (#batch)
    '''
    batch_size = z.shape[0]

    _z = z.clone().clamp(-math.pi, math.pi)
    _y = y.clone().clamp(-math.pi, math.pi)
    _x = x.clone().clamp(-math.pi, math.pi)

    ones = torch.ones(batch_size).type(torch.FloatTensor)
    zeros = torch.zeros(batch_size).type(torch.FloatTensor)

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    # shape: (#batch,3)
    rotz_mat_r1 = torch.stack((cosz, -sinz, zeros), dim=1)
    rotz_mat_r2 = torch.stack((sinz, cosz, zeros), dim=1)
    rotz_mat_r3 = torch.stack((zeros, zeros, ones), dim=1)
    # shape: (#batch,3,3)
    rotz_mat = torch.stack((rotz_mat_r1, rotz_mat_r2, rotz_mat_r3), dim=1)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    roty_mat_r1 = torch.stack((cosy, zeros, siny), dim=1)
    roty_mat_r2 = torch.stack((zeros, ones, zeros), dim=1)
    roty_mat_r3 = torch.stack((-siny, zeros, cosy), dim=1)
    roty_mat = torch.stack((roty_mat_r1, roty_mat_r2, roty_mat_r3), dim=1)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    rotx_mat_r1 = torch.stack((ones, zeros, zeros), dim=1)
    rotx_mat_r2 = torch.stack((zeros, cosx, -sinx), dim=1)
    rotx_mat_r3 = torch.stack((zeros, sinx, cosx), dim=1)
    rotx_mat = torch.stack((rotx_mat_r1, rotx_mat_r2, rotx_mat_r3), dim=1)

    # shape: (#batch,3,3)
    rot_mat = torch.matmul(rotz_mat, torch.matmul(roty_mat, rotx_mat))

    return rot_mat


def pose_vec2mat(vec):
    # TODO:pose vec 2 mat
    # input shape of vec: (#batch, 6)
    # shape: (#batch,3)
    translate_vec = vec[:, :3]
    rot_x = vec[:, 3]
    rot_y = vec[:, 4]
    rot_z = vec[:, 5]

    # shape: (#batch,3,3)
    rotation_mat = euler2mat(rot_z, rot_y, rot_x)

    # shape: (#batch,3,4)
    transform_mat = torch.cat((rotation_mat, translate_vec), dim=2)
    batch_size = vec.shape[0]
    # shape: (#batch,1,4)
    fill = torch.tensor([0, 0, 0, 1]).type(
        torch.FloatTensor).view(1, 4).repeat(batch_size, 1, 1)
    # shape: (#batch,4,4)
    transform_mat = torch.cat((transform_mat, fill), dim=1)
    return transform_mat



def meshgrid(batch, height, width, is_homogeneous=True):
    # TODO: meshgrid
    x = torch.ones(height).type(torch.FloatTensor).view(
        height, 1)*torch.linsapce(0, 1, width).view(1, width)
    y = torch.linspace(0, 1, height).view(height, 1) * \
        torch.ones(width).type(torch.FloatTensor).view(1, width)
    x = x*(width-1)
    y = y*(height-1)
    if is_homogeneous:
        ones = torch.ones(height, width)
        coords = torch.stack((x, y, ones), dim=0)  # shape: 3,h,w
    else:
        coords = torch.stack((x, y), dim=0)  # shape: 2,h,w
    coords = torch.repeat(batch, 1, 1, 1)
    # shape: #batch, 2 or 3, h, w
    return coords


def compute_rigid_flow(pose, depth, intrinsics, reverse_pose):
    # TODO: compute rigid flow
    '''Compute the rigid flow from src view to tgt view 

        input shapes:
            pose: #batch,6
            depth: #batch,h,w
            intrinsics: #batch,3,3
    '''
    batch_size, h, w = depth.shape

    pose_mat = pose_vec2mat(pose)
    if reverse_pose:
        pose_mat = torch.inverse(pose_mat)

    intrinsics_inv = torch.inverse(intrinsics)

    # shape: (#batch*h*w,3)
    src_coords = meshgrid(batch_size, h, w, True).permute(
        0, 2, 3, 1).view(batch_size*h*w, 3)
    # shape: (#batch*h*w,3)
    _depth = depth.clone().view(batch_size*h*w).repeat(3, 1).transpose(1, 0)
    # shape: (#batch*h*w,3)
    tgt_coords = _depth*torch.matmul(intrinsics_inv, src_coords)

    ones = torch.ones(batch_size).type(torch.FloatTensor).view(batch_size, 1)
    # shape: (#batch*h*w,4)
    tgt_coords = torch.cat((tgt_coords, ones), dim=1)

    # shape: (#batch*h*w,3)
    tgt_coords = torch.matmul(pose_mat, tgt_coords)[:, :3]

    # shape: (#batch*h*w,3)
    tgt_coords = torch.matmul(intrinsics, tgt_coords)

    # shape: (#batch*h*w,3)
    rigid_flow = tgt_coords-src_coords
    # shape: (#batch,2,h,w)
    rigid_flow = rigid_flow[:, :2].view(
        batch_size, h, w, 2).permute(batch_size, 2, h, w)
    return rigid_flow


def flow_warp(src_img, src2tgt_flow):
    # TODO: flow warp
    flow_field = src2tgt_flow.clone().permute(0, 2, 3, 1)
    tgt_img = F.grid_sample(src_img, flow_field)
    return tgt_img
