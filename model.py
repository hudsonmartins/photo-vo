
import torch
import numpy as np
from torch import nn
import gluefactory as gf
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from utils import get_patches, get_sorted_matches, normalize_image
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

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Extracted from: https://github.com/nianticlabs/monodepth2/blob/master/networks/resnet_encoder.py#L17
    """
    def __init__(self, block, layers, num_input_images=2):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ImagePairEncoder(nn.Module):
    def __init__(self):
        super(ImagePairEncoder, self).__init__()

        self.channels = np.array([64, 64, 128, 256, 512])
        block_type = models.resnet.BasicBlock
        layers = [2, 2, 2, 2]
        self.encoder = ResNetMultiImageInput(block_type, layers, num_input_images=2)
        resnet18 = model_zoo.load_url(models.resnet.model_urls['resnet18'])
        resnet18['conv1.weight'] = torch.cat([resnet18['conv1.weight']] * 2, 1)/2
        self.encoder.load_state_dict(resnet18)
    
    def forward(self, data):
        im0 = data['view0']['image']
        im1 = data['view1']['image']
        im0 = normalize_image(im0)
        im1 = normalize_image(im1)
        images = torch.cat([im0, im1], dim=1)
        x = self.encoder.conv1(images)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        return x

class MotionEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = 2*3 + config.photo_vo.model.dim_patch_emb #num_views * num_channels + n encodings
        
    def forward(self, data):
        pass


class PhotoVoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.penc = PatchEncoder(config)
        self.imgenc = ImagePairEncoder()
        self.matcher = gf.models.get_model(config.features_model.name)(config.features_model)
        #self.motion_estimator = MotionEstimator(config)

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
        
        print('patch embs ', patch_embs.shape)
        
        # Encode images
        image_embs = self.imgenc(data)
        print('image embs ', image_embs.shape)


        output = None
        #output = self.motion_estimator({**data, **feats})
        return output

    
def get_photo_vo_model(config):
    return PhotoVoModel(config)