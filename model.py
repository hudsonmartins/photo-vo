
import torch
from torch import nn
import gluefactory as gf
from torchvision.models import resnet18
from typing import Callable, List, Optional
from utils import get_patches, get_sorted_matches
from loss import photometric_loss, pose_error


class PatchEncoder(nn.Module):
    def __init__(self, config):
        '''
        Patch encoder that encodes the patches into a feature vector
        assumes that the patches are sorted by confidence
        '''
        super().__init__()
        patch_dim = 3 * config.photo_vo.model.patch_size ** 2
        self.patch_emb = nn.Sequential(nn.Flatten(start_dim=2, end_dim=4),
                                       nn.LayerNorm(patch_dim),
                                       nn.Linear(patch_dim, config.photo_vo.model.dim_patch_emb),
                                       nn.LayerNorm(config.photo_vo.model.dim_patch_emb))
        
    def forward(self, data):
        patches = torch.cat([data['view0']['patches'], data['view1']['patches']], dim=1)
        return self.patch_emb(patches)


class MotionEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        input_size = 2*3 + config.photo_vo.model.dim_patch_emb #num_views * num_channels + n encodings
        self.resnet.conv1 = nn.Conv2d(input_size, 64, kernel_size=3, stride=1, bias=False)
        n_emb = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.pose_est = nn.Sequential(nn.Linear(n_emb, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 6))
        
    def forward(self, data):
        im0 = data['view0']['image']
        im1 = data['view1']['image']
        images = torch.cat([im0, im1], dim=1)
        embs = self.resnet(images)
        egomotion = self.pose_est(embs)
        output = {**data, 'egomotion': egomotion}
        return output


class PhotoVoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.penc = PatchEncoder(config)
        self.motion_estimator = MotionEstimator(config)
        self.matcher = gf.models.get_model(config.features_model.name)(config.features_model)
    
    def forward(self, data):
        if 'view0' in data.keys() and 'view1' in data.keys():
            size0 = data['view0'].get('image_size')
            size1 = data['view1'].get('image_size')
        feats = self.matcher(data)
        kpts0, kpts1 = feats['keypoints0'], feats['keypoints1']

        sorted_matches = get_sorted_matches(feats)
        b_valid_matches = []
        for b in range(sorted_matches.size(0)):
            valid_matches = []
            for m in sorted_matches[b]:
                if m[1] > -1:
                    valid_matches.append([m[0], m[1]])
            b_valid_matches.append(valid_matches)

        # Extracting matches for m0 and m1
        m0 = [m[0].long() for matches in b_valid_matches for m in matches]
        m1 = [m[1].long() for matches in b_valid_matches for m in matches]
                
        # Indexing kpts0 and kpts1 with valid matches
        kpts0_valid = kpts0[:,m0,:]
        kpts1_valid = kpts1[:,m1,:]
        scores0_valid = feats['matching_scores0'][:,m0]
        scores1_valid = feats['matching_scores1'][:,m1]
        
        # Fill the invalid kpts with nan
        kpts0 = torch.cat([kpts0_valid, torch.full((kpts0_valid.size(0), kpts0.size(1)-kpts0_valid.size(1), kpts0_valid.size(2)), float('nan'), dtype=kpts0_valid.dtype, device=kpts0_valid.device)], dim=1)
        kpts1 = torch.cat([kpts1_valid, torch.full((kpts1_valid.size(0), kpts1.size(1)-kpts1_valid.size(1), kpts1_valid.size(2)), float('nan'), dtype=kpts1_valid.dtype, device=kpts1_valid.device)], dim=1)
        
        # Extract patches
        patches0 = get_patches(data['view0']['image'], kpts0, self.config.photo_vo.model.patch_size)
        patches1 = get_patches(data['view1']['image'], kpts1, self.config.photo_vo.model.patch_size)
        
        # Fill the invalid patches with -1
        patches0 = torch.nan_to_num(patches0, nan=-1.0)
        patches1 = torch.nan_to_num(patches1, nan=-1.0)
        data['view0']['patches'] = patches0
        data['view1']['patches'] = patches1
        # Encode patches
        patch_embs= self.penc(data)

        # Concat with scores
        scores0 = torch.cat([scores0_valid, torch.full((scores0_valid.size(0), kpts0.size(1)-scores0_valid.size(1)), -1.0, dtype=scores0_valid.dtype, device=scores0_valid.device)], dim=1)
        scores1 = torch.cat([scores1_valid, torch.full((scores1_valid.size(0), kpts1.size(1)-scores1_valid.size(1)), -1.0, dtype=scores1_valid.dtype, device=scores1_valid.device)], dim=1)
        scores = torch.cat([scores0, scores1], dim=1)
        patch_embs = torch.cat([patch_embs, scores.unsqueeze(-1)], dim=-1)
        
        output = None
        #output = self.motion_estimator({**data, **feats})
        return output

    
def get_photo_vo_model(config):
    return PhotoVoModel(config)