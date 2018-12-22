import torch
import torch.nn.functional as F
# from torch.autograd import Variable
import math
# from torchvision.transforms import Resize


# def resize_2d(img, size):
#     # Support resizin on GPU
#     return (F.adaptive_avg_pool2d(Variable(img, volatile=True), size)).data
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


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
            scaled_img = torch.nn.functional.interpolate(img, size=(nh, nw))
            scaled_imgs.append(scaled_img)
    # shape: #scales, # batch, #chnl, h, w
    return scaled_imgs


def L2_norm(x, dim, keep_dims=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset,
                         dim=dim, keepdim=keep_dims)
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
    # sigma_x = avepooling2d((x-mu_x)**2)
    # sigma_y = avepooling2d((y-mu_y)**2)
    # sigma_xy = avepooling2d((x-mu_x)*(y-mu_y))
    sigma_x = avepooling2d(x**2)-mu_x**2
    sigma_y = avepooling2d(y**2)-mu_y**2
    sigma_xy = avepooling2d(x*y)-mu_x*mu_y
    k1_square = 0.01**2
    k2_square = 0.03**2
    # L_square = 255**2
    L_square = 1
    SSIM_n = (2*mu_x*mu_y+k1_square*L_square)*(2*sigma_xy+k2_square*L_square)
    SSIM_d = (mu_x**2+mu_y**2+k1_square*L_square) * \
        (sigma_x+sigma_y+k2_square*L_square)
    SSIM = SSIM_n/SSIM_d
    return torch.clamp((1-SSIM)/2, 0, 1)


def gradient_x(img):
    return img[:, :, :, :-1]-img[:, :, :, 1:]


def gradient_y(img):
    return img[:, :, :-1, :]-img[:, :, 1:, :]


# def residual_flow(intrinsics, T, Depth, pt):
#     # BETTER to use KTDK'pt-pt as matrix multiple or by formulation as equations?
#     pc = torch.tensor([(pt[0]-intrinsics.cx)*Depth/intrinsics.fx,
#                        (pt[1]-intrinsics.cy)*Depth/intrinsics.fy,
#                        Depth,
#                        1],
#                       requires_grad=True).to(device)
#     pc_n = torch.matmul(T, pc)
#     pt_n = torch.tensor([intrinsics.fx*pc_n[0]/pc_n[2]+intrinsics.cx,
#                          intrinsics.fy*pc_n[1]/pc_n[2]+intrinsics.cy],
#                         requires_grad=True)
#     return pt_n-pt


def compute_multi_scale_intrinsics(intrinsics, num_scales):

    batch_size = intrinsics.shape[0]
    multi_scale_intrinsices = []
    for s in range(num_scales):
        fx = intrinsics[:, 0, 0]/(2**s)
        fy = intrinsics[:, 1, 1]/(2**s)
        cx = intrinsics[:, 0, 2]/(2**s)
        cy = intrinsics[:, 1, 2]/(2**s)
        zeros = torch.zeros(batch_size).float().to(device)
        r1 = torch.stack([fx, zeros, cx], dim=1)  # shape: batch_size,3
        r2 = torch.stack([zeros, fy, cy], dim=1)  # shape: batch_size,3
        # shape: batch_size,3
        r3 = torch.tensor([0., 0., 1.]).float().view(
            1, 3).repeat(batch_size, 1).to(device)
        # concat along the spatial row dimension
        scale_instrinsics = torch.stack([r1, r2, r3], dim=1)
        multi_scale_intrinsices.append(
            scale_instrinsics)  # shape: num_scale,bs,3,3
    multi_scale_intrinsices = torch.stack(multi_scale_intrinsices, dim=1)
    return multi_scale_intrinsices


def euler2mat(z, y, x):
    global device
    # TODO: eular2mat
    '''
    input shapes of z,y,x all are: (#batch)
    '''
    batch_size = z.shape[0]

    _z = z.clone().clamp(-math.pi, math.pi)
    _y = y.clone().clamp(-math.pi, math.pi)
    _x = x.clone().clamp(-math.pi, math.pi)

    ones = torch.ones(batch_size).float().to(device)
    zeros = torch.zeros(batch_size).float().to(device)

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
    global device
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
    transform_mat = torch.cat(
        (rotation_mat, translate_vec.unsqueeze(2)), dim=2)
    batch_size = vec.shape[0]
    # shape: (#batch,1,4)
    fill = torch.tensor([0, 0, 0, 1]).type(
        torch.FloatTensor).view(1, 4).repeat(batch_size, 1, 1).to(device)
    # shape: (#batch,4,4)
    transform_mat = torch.cat((transform_mat, fill), dim=1)
    return transform_mat


def meshgrid(height, width, is_homogeneous=True):
    global device
    x = torch.ones(height).float().view(height, 1).to(device)
    # shape : (h,w)
    x = torch.matmul(x, torch.linspace(0, 1, width).view(1, width).to(device))

    y = torch.linspace(0, 1, height).view(height, 1).to(device)
    # shape : (h,w)
    y = torch.matmul(y, torch.ones(width).float().view(1, width).to(device))

    x = x*(width-1)
    y = y*(height-1)

    if is_homogeneous:
        ones = torch.ones(height, width).float().to(device)
        coords = torch.stack((x, y, ones), dim=2)  # shape: h,w,3
    else:
        coords = torch.stack((x, y), dim=2)  # shape: h,w,2

    # shape:  (h, w, 2 or 3)
    return coords


def compute_rigid_flow(pose, depth, intrinsics, reverse_pose):
    global device
    '''Compute the rigid flow from src view to tgt view 

        input shapes:
            pose: #batch,6
            depth: #batch,h,w
            intrinsics: #batch,3,3
    '''
    batch_size, h, w = depth.shape
    # shape: (#batch,4,4)
    pose_mat = pose_vec2mat(pose)
    if reverse_pose:
        pose_mat = torch.inverse(pose_mat)

    # shape: (#batch,1,3,3)
    intrinsics_inv = torch.inverse(intrinsics).unsqueeze_(1)

    # shape: (#h*w,3,1)
    src_coords = meshgrid(h, w, True).contiguous().view(h*w, 3, 1)

    # shape: matmul( (#batch,1,3,3) ,(h*w,3,1)) = (#batch,h*w,3,1)
    tgt_coords = torch.matmul(intrinsics_inv, src_coords)

    # shape: (#batch, h*w,3,1)
    _depth = depth.view(batch_size, h*w).repeat(3, 1,
                                                        1).permute(1, 2, 0).unsqueeze_(3)

    # point-wise multiplication : shape: (# batch, h*w,3,1)
    tgt_coords = _depth * tgt_coords

    ones = torch.ones(batch_size, h*w, 1, 1).float().to(device)

    # shape: (#batch, h*w,4,1)
    tgt_coords = torch.cat((tgt_coords, ones), dim=2)

    # shape: (#batch,h*w,4, 4)
    pose_mat = pose_mat.repeat(h*w, 1, 1, 1).transpose(1, 0)

    # shape: matmul((#batch,h*w,4,4),(#batch,h*w,4,1)) = (#batch,h*w,4,1) -> (#batch,h*w,3,1)
    tgt_coords = torch.matmul(pose_mat, tgt_coords)[:, :, :3, :]

    # shape: (#batch,h*w,3, 3)
    intrinsics = intrinsics.repeat(h*w, 1, 1, 1).transpose(1, 0)

    # shape: matmul((#batch,h*w,3,3),(#batch,h*w,3,1)) = (#batch,h*w,3,1)
    tgt_coords = torch.matmul(intrinsics, tgt_coords)

    # shape: (#batch,h*w,3,1)
    src_coords = src_coords.repeat(batch_size, 1, 1, 1)
    # shape: (#batch,h*w,3,1)
    # rigid_flow = tgt_coords-src_coords
    # shape: (#batch,h*w,2)
    tgt_depth = tgt_coords[:, :, 2, :].clone().repeat(1,1,2) # require grad but also require modify (repeat) here
    # shape: (#batch,h*w,2)
    tgt_coords = tgt_coords[:, :, :2, :].squeeze_(-1)/tgt_depth
    
    # shape: (#batch, h*w,2)
    # from IPython import embed
    # embed()
    normalizer = torch.tensor([(2./w),(2./h)]).repeat(batch_size,h*w,1).float().to(device)
    tgt_coords = tgt_coords*normalizer-1.
    tgt_coords = tgt_coords.contiguous().view(batch_size,h,w,2).permute(0,3,1,2)
    # shape: (#batch,2,h,w)
    # rigid_flow = rigid_flow[:, :, :2, :].contiguous().view(
    #     batch_size, h, w, 2, 1).squeeze_(-1).permute(0, 3, 1, 2)
    return tgt_coords


def flow_warp(src_img, src2tgt_flow):
    # TODO: flow warp
    flow_field = src2tgt_flow.permute(0, 2, 3, 1)
    tgt_img = F.grid_sample(src_img, flow_field)
    return tgt_img
