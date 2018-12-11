import torch
import torch.nn.functional as F
from torch.autograd import Variable

# TODO: why we need non homogeneous?

def resize_2d(img,size):
    # Support resizin on GPU
    return (F.adaptive_avg_pool2d(Variable(img,volatile=True),size)).data 

def scale_pyramid(img,num_scales):
    if img is None:
        return None
    else:
        scaled_imgs = [img]
        #TODO: Assume the shape of image is [#channels, #rows, #cols ]
        h,w = img.shape[1:3]
        for i in range(num_scales-1):   
            ratio = 2**(i+1)
            nh = int(h/ratio)
            nw = int(w/ratio)
            scaled_img = resize_2d(img,(nh,nw))
            scaled_imgs.append(scaled_img)
    return scaled_imgs


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

    avepooling2d = torch.nn.AvgPool2d(3, stride=1, padding=[1,1])
    mu_x = avepooling2d(x)
    mu_y = avepooling2d(y)
    sigma_x = avepooling2d((x-mu_x)**2) 
    sigma_y = avepooling2d((y-mu_y)**2)
    sigma_xy = avepooling2d((x-mu_x)*(y-mu_y))
    k1_square = 0.01**2
    k2_square = 0.03**2
    L_square = 255**2
    SSIM_n = (2*mu_x*mu_y+k1_square*L_square)*(2*sigma_xy+k2_square*L_square)
    SSIM_d = (mu_x**2+mu_y**2+k1_square*L_square)*(sigma_x+sigma_y+k2_square*L_square)
    SSIM = SSIM_n/SSIM_d
    return torch.clamp((1-SSIM)/2,0,1)


def gradient_x(img):
    return img[:,:,:,:-1]-img[:,:,:,1:]

def gradient_y(img):
    return img[:,:,:-1,:]-img[:,:,1:,:]

def residual_flow(intrinsics,T,Depth,pt):
    # BETTER to use KTDK'pt-pt as matrix multiple or by formulation as equations?
    pc = torch.tensor([(pt[0]-intrinsics.cx)*Depth/intrinsics.fx,
          (pt[1]-intrinsics.cy)*Depth/intrinsics.fy,
          Depth,
          1], 
          requires_grad=True)
    pc_n = torch.matmul(T,pc)
    pt_n = torch.tensor([intrinsics.fx*pc_n[0]/pc_n[2]+intrinsics.cx,
                        intrinsics.fy*pc_n[1]/pc_n[2]+intrinsics.cy],
                        requires_grad=True)
    return pt_n-pt

def compute_rigid_flow(pose,depth,intrinsic,reverse_pose):
    #TODO: compute rigid flow
    

def bilinear_sampler():
    # TODO: bilinear sample
    pass

def flow_warp():
    # TODO: flow warp
    pass

def euler2mat(z,y,x):
    # TODO: eular2mat
    pass

def pose_vec2mat(vec):
    # TODO:pose vec 2 mat
    pass

def pixel2cam(pixel_coords,depth,intrinsics):
    # TODO: pixel2cam
    pass

def cam2pixel(cam_coords,intrinsics):
    # TODO: cam2pixel
    pass

def meshgrid():
    # TODO: meshgrid
    pass



    