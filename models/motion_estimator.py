from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

class MotionEstimator(nn.Module):
    
    default_config = {
    
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, bias=False)
        n_feats = self.resnet.fc.in_features
        kpts_matches_size = config['n_kpts']+3
        self.resnet.fc = nn.Linear(n_feats, kpts_matches_size)       
        #TODO: we should test a more efficient architecture
        self.pose_est = nn.Sequential(nn.Linear(kpts_matches_size*(1+kpts_matches_size), 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 6))
        
    
    def normalize_keypoints(self, kpts, image_shape):
        """ Normalize keypoints locations based on image image_shape"""
        _, _, height, width = image_shape
        one = kpts.new_tensor(1)
        size = torch.stack([one*width, one*height])[None]
        center = size / 2
        scaling = size.max(1, keepdim=True).values * 0.7
        return (kpts - center[:, None, :]) / scaling[:, None, :]


    def concat_matches_kpts(self, matches, kpts0, kpts1):
        #creates a keypoint for each batch containing the dustbin
        dustbin0 = torch.add(torch.zeros((matches.shape[0], 1, 2)), -1).to(matches.device)
        kpts0_dustbin = torch.cat((kpts0, dustbin0), dim=1)
        dustbin1 = torch.add(torch.zeros((matches.shape[0], 1+kpts1.shape[2], 2)), -1).to(matches.device)
        kpts1_dustbin = torch.cat((kpts1, dustbin1), dim=1)
        matches_kpts = torch.cat((matches, kpts0_dustbin), dim=2)
        matches_kpts = torch.cat((matches_kpts, kpts1_dustbin.transpose(1,2)), dim=1)
        matches_kpts = matches_kpts.unsqueeze(1)
        return matches_kpts
    

    def forward(self, data):
        img0, img1 = data['image0'], data['image1']
        matches = data['matches']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        kpts0 = self.normalize_keypoints(kpts0, img0.shape)
        kpts1 = self.normalize_keypoints(kpts1, img1.shape)
        matches_kpts = self.concat_matches_kpts(matches, kpts0, kpts1)
        input = torch.cat([img0, img1], dim=1)
        resnet_out = self.resnet(input)

        output = self.pose_est(torch.cat([resnet_out, matches_kpts.flatten(1)], dim=1))
        return output
        
