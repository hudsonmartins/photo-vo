
import torch
from torch import nn
import gluefactory as gf
from torchvision.models import resnet18


class PatchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc = resnet18(pretrained=True)
        input_size = config.features_model.extractor.patch_size*2*3 #num_patches * num_views * num_channels
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
        print('n embs ', embs.shape)
        egomotion = self.pose_est(embs)
        output = {**data, 'egomotion': egomotion}
        return output


class PhotoVoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.motion_estimator = MotionEstimator(config)
        self.matcher = gf.models.get_model(config.features_model.name)(config.features_model)
    
    def forward(self, data):
        pred1 = self.matcher(data)
        output = self.motion_estimator({**data, **pred1})
        return output

    
def get_photo_vo_model(config):
    return PhotoVoModel(config)