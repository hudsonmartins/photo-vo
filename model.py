import torch
from torch import nn
import gluefactory as gf
from gluefactory.geometry.wrappers import Pose
from timesformer.models.vit import VisionTransformer
from model_tsformer import build_model as tsformer
from functools import partial

from utils import get_patches, get_sorted_matches, matrix_to_euler_angles
from loss import pose_error


class PatchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_dim = 3 * config.photo_vo.model.patch_size ** 2
        self.dim_emb = config.photo_vo.model.dim_emb
        
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, self.dim_emb),
            nn.LayerNorm(self.dim_emb)
        )
        
        # positional embedding from (x, y)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, self.dim_emb),
            nn.ReLU(),
            nn.Linear(self.dim_emb, self.dim_emb)
        )
        
        # Score fusion
        self.score_fc = nn.Sequential(
            nn.Linear(self.dim_emb + 1, self.dim_emb),
            nn.ReLU(),
            nn.LayerNorm(self.dim_emb)
        )

    def forward(self, data):
        patches0 = data['view0']['patches']  # (B, N, 3, H, W)
        patches1 = data['view1']['patches']
        coords0 = data['view0']['patches_coords']  # (B, N, 2)
        coords1 = data['view1']['patches_coords']
        scores0 = data['view0']['scores']  # (B, N)
        scores1 = data['view1']['scores']

        patches = torch.cat([patches0, patches1], dim=1)  # (B, 2N, 3, H, W)
        coords = torch.cat([coords0, coords1], dim=1)     # (B, 2N, 2)
        scores = torch.cat([scores0, scores1], dim=1)     # (B, 2N)

        B, N, C, H, W = patches.shape
        flat_patches = patches.view(B, N, -1)  # (B, 2N, 3*H*W)
        patch_feats = self.patch_proj(flat_patches)  # (B, 2N, D)

        pos_emb = self.pos_mlp(coords)  # (B, 2N, D)
        patch_feats = patch_feats + pos_emb

        fused = torch.cat([patch_feats, scores.unsqueeze(-1)], dim=-1)
        return self.score_fc(fused)  # (B, 2N, D)

    
class ImagePairEncoder(nn.Module):
    def __init__(self, config):
        super(ImagePairEncoder, self).__init__()
        model_args = {
            "window_size": 2,  # number of frames in window
            "pretrained_ViT": True,  # load weights from pre-trained ViT
            "checkpoint_path": "checkpoints_tsformer/",  # path to save checkpoint
            "checkpoint": None,  # checkpoint
            "optimizer": "Adam",  # optimizer [Adam, SGD, Adagrad, RAdam]
            "lr": 1e-5,  # learning rate
            "momentum": 0.9,  # SGD momentum
            "weight_decay": 1e-4,  # SGD momentum
        }

        model_params = {
            "dim": 384,
            "image_size": (640, 640), 
            "patch_size": 16,
            "attention_type": 'divided_space_time',  # ['divided_space_time', 'space_only','joint_space_time', 'time_only']
            "num_frames": model_args["window_size"],
            "num_classes": 6 * (model_args["window_size"] - 1),  # 6 DoF for each frame
            "depth": 12,
            "heads": 6,
            "dim_head": 64,
            "attn_dropout": 0.1,
            "ff_dropout": 0.1,
            "time_only": False,
        }
        self.vit = VisionTransformer(img_size=(640,640),
                            num_classes=model_params["num_classes"],
                            patch_size=model_params["patch_size"],
                            embed_dim=model_params["dim"],
                            depth=model_params["depth"],
                            num_heads=model_params["heads"],
                            mlp_ratio=4,
                            qkv_bias=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6),
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            drop_path_rate=0.1,
                            num_frames=model_params["num_frames"],
                            attention_type=model_params["attention_type"])
        checkpoint = torch.load('weights/tsformer-vo.tar', map_location=torch.device("cuda"))
        self.vit.load_state_dict(checkpoint['model'])
        self.vit.head = torch.nn.Identity() #remove last linear layer
        
    def forward(self, data):
        im0 = torch.clamp(data['view0']['image'], 0, 1)
        im1 = torch.clamp(data['view1']['image'], 0, 1)
        im0 = im0.unsqueeze(1)
        im1 = im1.unsqueeze(1)
        x = torch.cat([im0, im1], dim=1)
        x = x.transpose(1, 2)
        x = self.vit(x)
        return x

class MotionEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.photo_vo.model.dim_emb, num_heads=8
        )
        self.fc = nn.Linear(config.photo_vo.model.dim_emb, 6)

    def forward(self, image_embs, patch_embs):
        query = image_embs.unsqueeze(0)
        key = value = patch_embs.permute(1, 0, 2)  # Swap batch and sequence dimensions
        fused, _ = self.cross_attn(query, key, value)        
        fused = fused.squeeze(0)
        
        return self.fc(fused)


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
        # Encode full-frame images
        image_embs = self.imgenc(data)
        self.features = {}

        # Feature extraction and matching
        self.features = self.matcher(data)
        kpts0, kpts1 = self.features['keypoints0'], self.features['keypoints1']
        scores0, scores1 = self.features['matching_scores0'], self.features['matching_scores1']
        sorted_matches = get_sorted_matches(self.features)
        sorted_matches = sorted_matches[:, :self.config.photo_vo.model.num_matches]

        # Initialize containers
        kpts0_valid, kpts1_valid = None, None
        scores0_valid, scores1_valid = None, None

        for b in range(sorted_matches.size(0)):
            b_kpts0_valid, b_kpts1_valid = None, None
            b_scores0_valid, b_scores1_valid = None, None

            for m in sorted_matches[b]:
                if m[1] > -1:
                    # Valid match
                    i0, i1 = m[0].long(), m[1].long()
                    b_kpts0_valid = kpts0[b][i0].unsqueeze(0) if b_kpts0_valid is None else torch.cat([b_kpts0_valid, kpts0[b][i0].unsqueeze(0)], dim=0)
                    b_kpts1_valid = kpts1[b][i1].unsqueeze(0) if b_kpts1_valid is None else torch.cat([b_kpts1_valid, kpts1[b][i1].unsqueeze(0)], dim=0)
                    b_scores0_valid = scores0[b][i0].unsqueeze(0) if b_scores0_valid is None else torch.cat([b_scores0_valid, scores0[b][i0].unsqueeze(0)], dim=0)
                    b_scores1_valid = scores1[b][i1].unsqueeze(0) if b_scores1_valid is None else torch.cat([b_scores1_valid, scores1[b][i1].unsqueeze(0)], dim=0)
                else:
                    # Invalid match -> fill with -1s and zeros
                    invalid_kpts = torch.full((1, kpts0.size(2)), -1.0, dtype=kpts0.dtype, device=kpts0.device)
                    zero_score = torch.zeros(1, dtype=scores0.dtype, device=scores0.device)
                    b_kpts0_valid = invalid_kpts if b_kpts0_valid is None else torch.cat([b_kpts0_valid, invalid_kpts], dim=0)
                    b_kpts1_valid = invalid_kpts if b_kpts1_valid is None else torch.cat([b_kpts1_valid, invalid_kpts], dim=0)
                    b_scores0_valid = zero_score if b_scores0_valid is None else torch.cat([b_scores0_valid, zero_score], dim=0)
                    b_scores1_valid = zero_score if b_scores1_valid is None else torch.cat([b_scores1_valid, zero_score], dim=0)

            # Stack batch-wise
            kpts0_valid = b_kpts0_valid.unsqueeze(0) if kpts0_valid is None else torch.cat([kpts0_valid, b_kpts0_valid.unsqueeze(0)], dim=0)
            kpts1_valid = b_kpts1_valid.unsqueeze(0) if kpts1_valid is None else torch.cat([kpts1_valid, b_kpts1_valid.unsqueeze(0)], dim=0)
            scores0_valid = b_scores0_valid.unsqueeze(0) if scores0_valid is None else torch.cat([scores0_valid, b_scores0_valid.unsqueeze(0)], dim=0)
            scores1_valid = b_scores1_valid.unsqueeze(0) if scores1_valid is None else torch.cat([scores1_valid, b_scores1_valid.unsqueeze(0)], dim=0)

        # Extract local patches from both views using valid keypoints
        patches0 = get_patches(data['view0']['image'], kpts0_valid, self.config.photo_vo.model.patch_size)
        patches1 = get_patches(data['view1']['image'], kpts1_valid, self.config.photo_vo.model.patch_size)

        # Store patch-related info in data
        data['view0']['patches'] = patches0
        data['view1']['patches'] = patches1
        data['view0']['patches_coords'] = kpts0_valid
        data['view1']['patches_coords'] = kpts1_valid
        data['view0']['scores'] = scores0_valid
        data['view1']['scores'] = scores1_valid

        patch_embs = self.penc(data) 
        output = self.motion_estimator(image_embs, patch_embs)
        data['pred_vo'] = output

        return {**data, **self.features}


    def loss(self, data):
        pred = data['pred_vo']
        gt_R = data['T_0to1'].R
        gt_t = data['T_0to1'].t
        gt = torch.cat((matrix_to_euler_angles(gt_R, "ZYX"), gt_t), dim=1)
        pe = pose_error(gt, pred)
        data['gt_vo'] = gt
        loss = {'pose_error': pe, 'total': pe}
        
        return loss, data    
    
def get_photo_vo_model(config):
    return PhotoVoModel(config)    