
import torch
import numpy as np
from torch import nn
import gluefactory as gf
from gluefactory.geometry.wrappers import Pose

import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from utils import get_patches, get_sorted_matches, normalize_image, get_kpts_projection, matrix_to_euler_angles, euler_angles_to_matrix
from loss import patches_photometric_loss, pose_error


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
                                       nn.Linear(patch_dim, config.photo_vo.model.dim_image_emb-1),
                                       nn.LayerNorm(config.photo_vo.model.dim_image_emb-1))
        
    def forward(self, data):
        patches = torch.cat([data['view0']['patches'], data['view1']['patches']], dim=1)
        return self.patch_emb(patches)
    

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Extracted from: https://github.com/nianticlabs/monodepth2/blob/master/networks/resnet_encoder.py#L17
    """
    def __init__(self, block, layers, config, num_input_images=2):
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
        self.layer4 = self._make_layer(block, config.photo_vo.model.dim_image_emb, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ImagePairEncoder(nn.Module):
    def __init__(self, config):
        super(ImagePairEncoder, self).__init__()
        
        self.channels = np.array([64, 64, 128, 256, config.photo_vo.model.dim_image_emb])
        block_type = models.resnet.BasicBlock
        layers = [2, 2, 2, 2]
        self.encoder = ResNetMultiImageInput(block_type, layers, config, num_input_images=2)
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
        self.flatten = nn.Sequential(nn.Flatten(start_dim=2, end_dim=3),)
        self.decoder_layers = [config.photo_vo.model.dim_image_emb, 128, 64, 32, 6]

    def forward(self, image_embs, patch_embs):
        patch_embs = patch_embs.permute(0, 2, 1)
        x = torch.cat([self.flatten(image_embs), patch_embs], dim=2)
        for i in range(len(self.decoder_layers)-1):
            x = nn.Conv1d(self.decoder_layers[i], self.decoder_layers[i+1], 1)(x)
            x = nn.ReLU()(x)
        return torch.mean(x, dim=2)


class PhotoVoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.imgenc = ImagePairEncoder(config)
        self.matcher = gf.models.get_model(config.features_model.name)(config.features_model)
        self.penc = PatchEncoder(config)
        self.motion_estimator = MotionEstimator(config)
        self.features = None

    def forward(self, data):
        # Encode images
        image_embs = self.imgenc(data)

        # Extract and match features
        self.features = self.matcher(data)

        kpts0, kpts1 = self.features['keypoints0'], self.features['keypoints1']

        sorted_matches = get_sorted_matches(self.features)
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
        scores0_valid = self.features['matching_scores0'][:,m0]
        scores1_valid = self.features['matching_scores1'][:,m1]
        
        # Fill the invalid kpts with nan
        kpts0 = torch.cat([kpts0_valid, torch.full((kpts0_valid.size(0), kpts0.size(1)-kpts0_valid.size(1), kpts0_valid.size(2)), float('nan'), dtype=kpts0_valid.dtype, device=kpts0_valid.device)], dim=1)
        kpts1 = torch.cat([kpts1_valid, torch.full((kpts1_valid.size(0), kpts1.size(1)-kpts1_valid.size(1), kpts1_valid.size(2)), float('nan'), dtype=kpts1_valid.dtype, device=kpts1_valid.device)], dim=1)
        
        # Extract patches
        patches0 = get_patches(data['view0']['image'], kpts0, self.config.photo_vo.model.patch_size)
        patches1 = get_patches(data['view1']['image'], kpts1, self.config.photo_vo.model.patch_size)
        
        # Fill the invalid patches with -1
        patches0 = torch.nan_to_num(patches0, nan=-1.0)
        patches1 = torch.nan_to_num(patches1, nan=-1.0)
        data['view0']['patches_coords'] = kpts0
        data['view1']['patches_coords'] = kpts1
        data['view0']['patches'] = patches0
        data['view1']['patches'] = patches1
        # Encode patches
        patch_embs= self.penc(data)
        
        # Concat with scores
        scores0 = torch.cat([scores0_valid, torch.full((scores0_valid.size(0), kpts0.size(1)-scores0_valid.size(1)), -1.0, dtype=scores0_valid.dtype, device=scores0_valid.device)], dim=1)
        scores1 = torch.cat([scores1_valid, torch.full((scores1_valid.size(0), kpts1.size(1)-scores1_valid.size(1)), -1.0, dtype=scores1_valid.dtype, device=scores1_valid.device)], dim=1)
        scores = torch.cat([scores0, scores1], dim=1)
        patch_embs = torch.cat([patch_embs, scores.unsqueeze(-1)], dim=-1)
        
        output = self.motion_estimator(image_embs, patch_embs)
        data['pred_vo'] = output
        return {**data, **self.features}


    def loss(self, data):
        pred = data['pred_vo']
        kpts0, kpts1 = data['view0']['patches_coords'], data['view1']['patches_coords']
        depth0 = data["view0"].get("depth")
        depth1 = data["view1"].get("depth")
        camera0, camera1 = data["view0"]["camera"], data["view1"]["camera"]

        #Compute projections with predicted camera pose
        R_pred = euler_angles_to_matrix(pred[..., 3:], "XYZ")
        t_pred = pred[..., :3]
        T_0to1_pred = Pose.from_Rt(R_pred, t_pred)
        T_1to0_pred = T_0to1_pred.inv()
        kpts0_1 = get_kpts_projection(kpts0, depth0, camera0, camera1, T_0to1_pred)
        kpts1_0 = get_kpts_projection(kpts1, depth1, camera1, camera0, T_1to0_pred)
        
        patches0 = data['view0']['patches']
        patches1 = data['view1']['patches']

        patches0_1 = get_patches(data['view0']['image'], kpts0_1, self.config.photo_vo.model.patch_size)
        patches1_0 = get_patches(data['view1']['image'], kpts1_0, self.config.photo_vo.model.patch_size)
        patches0_1 = torch.nan_to_num(patches0_1, nan=-1.0)
        patches1_0 = torch.nan_to_num(patches1_0, nan=-1.0)

        pl0 = patches_photometric_loss(patches0, patches0_1)
        pl1 = patches_photometric_loss(patches1, patches1_0)
        pl = (pl0 + pl1)/2

        gt_R = data['T_0to1'].R
        gt_t = data['T_0to1'].t
        gt = torch.cat((gt_t, matrix_to_euler_angles(gt_R, "XYZ")), dim=1)
        pe = pose_error(gt, pred)

        match_losses, _ = self.matcher.loss(self.features, data)
        ml = match_losses['total']
        w1 = self.config.photo_vo.model.loss_weights.photometric
        w2 = self.config.photo_vo.model.loss_weights.pose
        w3 = self.config.photo_vo.model.loss_weights.match
        loss = {'photometric_loss': pl, 'pose_error': pe, 'match_loss': ml, 'total': w1*pl + w2*pe + w3*ml}
        return loss
        
    
def get_photo_vo_model(config):
    return PhotoVoModel(config)