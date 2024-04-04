import torch
from utils import matrix_to_euler_angles

def get_ssim(img0, img1):
    #Extracted from https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py#L91
    C1 = 0.01**2
    C2 = 0.03**2
    mean0 = torch.nn.functional.avg_pool2d(img0, 3, 1)
    mean1 = torch.nn.functional.avg_pool2d(img1, 3, 1)
    std0  = torch.nn.functional.avg_pool2d(img0 ** 2, 3, 1) - mean0 ** 2
    std1  = torch.nn.functional.avg_pool2d(img1 ** 2, 3, 1) - mean1 ** 2
    std01 = torch.nn.functional.avg_pool2d(img0 * img1 , 3, 1) - mean0 * mean1
    
    SSIM_n = (2 * mean0 * mean1 + C1) * (2 * std01 + C2)
    SSIM_d = (mean0 ** 2 + mean1 ** 2 + C1) * (std0 + std1 + C2)

    SSIM = SSIM_n / SSIM_d
    # Return the average SSIM across the batch
    return torch.mean(SSIM)


def photometric_loss(img0, img1):
    ssim = get_ssim(img0, img1)
    ssim_loss = torch.mean(torch.clamp((1 - ssim), 0, 1))
    l1 = torch.abs((img0 - img1))
    l1_loss = torch.mean(l1)
    alpha = 0.85 #Zhao et al
    pe = alpha * ssim_loss + (1-alpha) * l1_loss
    return pe


def patches_photometric_loss(patches0, patches1):
    batch_size = patches0.size(0)
    loss = 0
    for i in range(batch_size):
        loss += photometric_loss(patches0[i], patches1[i])
    return loss/batch_size


def pose_error(gt, pred):
    """Compute the pose error between true and predicted poses."""
    #matrix from R, t
    return torch.mean(torch.linalg.norm(pred - gt, dim=1, ord=2))


if __name__ == "__main__":
    img0 = torch.rand(8, 3, 256, 256)
    img1 = torch.rand(8, 3, 256, 256)
    print('img 0 vs img1 ', photometric_loss(img0, img1))
    print('img 0 vs img0 ', photometric_loss(img0, img0))
    print('img 1 vs img1 ', photometric_loss(img1, img1))