
import torch
from torch import nn
import gluefactory as gf
from torchvision.models import resnet18
from utils import get_patches, get_sorted_matches

class PatchEncoder(nn.Module):
    def __init__(self, config):
        '''
        Patch encoder that encodes the patches into a feature vector
        assumes that the patches are sorted by confidence
        '''
        super().__init__()
        self.enc = resnet18(pretrained=True)
        input_size = config.patch_encoder.model.patch_size*2*3 #num_patches * num_views * num_channels
        self.enc.conv1 = nn.Conv2d(input_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.enc.fc = nn.Identity()
        
    def forward(self, data):
        input = torch.cat([data["view0"]["patches"], data["view1"]["patches"]], dim=1)        
        return self.enc(input)
    

class MotionEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, bias=False)
        n_emb = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.pose_est = nn.Sequential(nn.Linear(n_emb, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 6))
        
    def forward(self, data):
        im0 = data["view0"]["image"]
        im1 = data["view1"]["image"]
        images = torch.cat([im0, im1], dim=1)
        embs = self.resnet(images)
        egomotion = self.pose_est(embs)
        output = {**data, 'egomotion': egomotion}
        return output


class PhotoVoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.penc = PatchEncoder(config)
        self.motion_estimator = MotionEstimator(config)
        self.matcher = gf.models.get_model(config.features_model.name)(config.features_model)
    
    def forward(self, data):
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
        # Fill the invalid kpts with nan
        kpts0 = torch.cat([kpts0_valid, torch.full((kpts0_valid.size(0), kpts0.size(1)-kpts0_valid.size(1), kpts0_valid.size(2)), 1, dtype=kpts0_valid.dtype, device=kpts0_valid.device)], dim=1)
        kpts1 = torch.cat([kpts1_valid, torch.full((kpts1_valid.size(0), kpts1.size(1)-kpts1_valid.size(1), kpts1_valid.size(2)), 1, dtype=kpts1_valid.dtype, device=kpts1_valid.device)], dim=1)
        
        patches0 = get_patches(data["view0"]["image"], kpts0)
        patches1 = get_patches(data["view1"]["image"], kpts1)

        data['view0']['patches'] = patches0
        data['view1']['patches'] = patches1

        output = self.motion_estimator({**data, **feats})
        return output

    
def get_photo_vo_model(config):
    return PhotoVoModel(config)