import torch
import numpy as np
from utils import matrix_to_euler_angles

def get_ssim(Ia, Ib):
    #based in https://github.com/mrharicot/monodepth/blob/master/monodepth_model.py#L91
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
    ssim_loss = torch.mean(torch.clamp((1 - ssim)/2, 0, 1))
    l1 = torch.abs((img0 - img1))
    l1_loss = torch.mean(l1)
    alpha = 0.85 #Zhao et al
    pe = alpha * ssim_loss + (1-alpha) * l1_loss
    return pe


def patches_photometric_loss(patches0, patches1, matches):
    batch_size = patches0.size(0)
    losses = []
    for i in range(batch_size):
        valid_matches = (matches[i] >= 0).nonzero(as_tuple=True)[0]
        for j in valid_matches:
            #Gives the maximum possible difference for patches where no warping is possible
            if(torch.sum(patches0[i, j]) == -1*patches0[i, j].numel()):
                ones = torch.ones_like(patches0[i, j])
                zeros = torch.zeros_like(patches1[i, matches[i][j]])
                losses += [photometric_loss(ones, zeros)]
            else:
                losses += [photometric_loss(patches0[i, j], patches1[i, matches[i][j]])]
    return torch.mean(torch.Tensor(losses))


def pose_error(pred, pose1, pose2):
    pose1_2 = torch.matmul(pose2, torch.inverse(pose1))
    R12_euler = matrix_to_euler_angles(pose1_2[..., :3, :3], convention="ZYX")
    gt = torch.cat((pose1_2[..., :3, 3], R12_euler), dim=1)
    return torch.mean(torch.linalg.norm(pred - gt, dim=1, ord=2))


def nll_loss(output, target, pos_weight=0.5, neg_weight=0.5):
    match_ids = (target[:, :-1, :-1] == 1).nonzero(as_tuple=True)
    unmatch1_ids = (target[:, :, -1] == 1).nonzero(as_tuple=True)
    unmatch2_ids = (target[:, -1, :] == 1).nonzero(as_tuple=True)
    
    pos_loss = -torch.mean(output[match_ids])
    neg_loss1 = -torch.mean(output[unmatch1_ids[0], unmatch1_ids[1], -1])
    neg_loss2 = -torch.mean(output[unmatch2_ids[0], -1, unmatch2_ids[1]])
    loss = pos_weight * pos_loss + neg_weight * (neg_loss1 + neg_loss2)
    
    return loss/target.size(0)


def nll_loss_le(output, target, unmatch_id, pos_weight=0.5, neg_weight=0.5):
    match_ids = ((target > 0) & (target < unmatch_id)).nonzero(as_tuple=False)
    match_ids = torch.transpose(match_ids, 1, 0)
    match_ids = torch.cat([match_ids, torch.unsqueeze(target[match_ids.numpy()], dim=0)], dim=0)
    match_probs = output[match_ids.numpy()]
    match_loss = -torch.sum(match_probs)/len(match_ids)

    unmatch_ids = (target == unmatch_id).nonzero(as_tuple=False)
    unmatch_ids = torch.transpose(unmatch_ids, 1, 0)
    unmatch_ids = torch.cat([unmatch_ids, torch.unsqueeze(target[unmatch_ids.numpy()], dim=0)], dim=0)
    unmatch_probs = output[unmatch_ids.numpy()]   
    unmatch_loss = -torch.sum(unmatch_probs)/len(unmatch_ids)

    loss = pos_weight * match_loss + neg_weight * unmatch_loss
    
    return loss/target.size(0)
