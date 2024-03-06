import torch
from utils import matrix_to_euler_angles

def get_ssim(Ia, Ib):
    #Extracted from https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py#L91
    C1 = 0.01**2
    C2 = 0.03**2
    mean_a = torch.nn.functional.avg_pool2d(Ia, 3, 1)
    mean_b = torch.nn.functional.avg_pool2d(Ib, 3, 1)
    std_a  = torch.nn.functional.avg_pool2d(Ia ** 2, 3, 1) - mean_a ** 2
    std_b  = torch.nn.functional.avg_pool2d(Ib ** 2, 3, 1) - mean_b ** 2
    std_ab = torch.nn.functional.avg_pool2d(Ia * Ib , 3, 1) - mean_a * mean_b
    
    SSIM_n = (2 * mean_a * mean_b + C1) * (2 * std_ab + C2)
    SSIM_d = (mean_a ** 2 + mean_b ** 2 + C1) * (std_a + std_b + C2)

    SSIM = SSIM_n / SSIM_d
    return SSIM


def photometric_loss(img0, img1):
    ssim = get_ssim(img0, img1)
    ssim_loss = torch.mean(torch.clamp((1 - ssim), 0, 1))
    l1 = torch.abs((img0 - img1))
    l1_loss = torch.mean(l1)
    alpha = 0.85 #Zhao et al
    pe = alpha * ssim_loss + (1-alpha) * l1_loss
    return pe


def pose_error(R_true, t_true, R_pred, t_pred):
    """Compute the pose error between true and predicted poses."""
    #matrix from R, t
    pred = torch.cat((t_pred, matrix_to_euler_angles(R_pred)), dim=1)
    gt = torch.cat((t_true, matrix_to_euler_angles(R_true)), dim=1)
    return torch.mean(torch.linalg.norm(pred - gt, dim=1, ord=2))


if __name__ == "__main__":
    img0 = torch.rand(8, 3, 256, 256)
    img1 = torch.rand(8, 3, 256, 256)
    print('img 0 vs img1 ', photometric_loss(img0, img1))
    print('img 0 vs img0 ', photometric_loss(img0, img0))
    print('img 1 vs img1 ', photometric_loss(img1, img1))