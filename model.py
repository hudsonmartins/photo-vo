import torch
from torch import nn
import gluefactory as gf
from timesformer.models.vit import VisionTransformer
from functools import partial
from collections import OrderedDict
from voflowres import VOFlowRes
from utils import get_patches, make_intrinsics_layer, get_sorted_matches


class PatchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_dim = 5 * config.photo_vo.model.patch_size ** 2
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
        patches0 = data['view0']['patches']  # (B, N, C, H, W)
        patches1 = data['view1']['patches']
        coords0 = data['view0']['patches_coords']  # (B, N, 2)
        coords1 = data['view1']['patches_coords']
        scores0 = data['view0']['scores']  # (B, N)
        scores1 = data['view1']['scores']
        
        patches = torch.cat([patches0, patches1], dim=1)  # (B, 2N, C, H, W)
        coords = torch.cat([coords0, coords1], dim=1)     # (B, 2N, 2)
        scores = torch.cat([scores0, scores1], dim=1)     # (B, 2N)

        B, N, C, H, W = patches.shape
        flat_patches = patches.view(B, N, -1)  # (B, 2N, C*H*W)
        patch_feats = self.patch_proj(flat_patches)  # (B, 2N, D)

        pos_emb = self.pos_mlp(coords)  # (B, 2N, D)
        patch_feats = patch_feats + pos_emb

        fused = torch.cat([patch_feats, scores.unsqueeze(-1)], dim=-1)
        return self.score_fc(fused)  # (B, 2N, D)
    
class ImagePairEncoder(nn.Module):
    def __init__(self, config):
        super(ImagePairEncoder, self).__init__()
        
        self.vit = VisionTransformer(img_size=config.vit.image_size,
                            num_classes=6,
                            patch_size=config.vit.patch_size,
                            embed_dim=config.vit.dim_emb,
                            depth=config.vit.depth,
                            num_heads=config.vit.heads,
                            mlp_ratio=4,
                            qkv_bias=True,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6),
                            drop_rate=0.,
                            attn_drop_rate=0.,
                            drop_path_rate=0.1,
                            num_frames=2,
                            attention_type='divided_space_time')
        
        self.adapter = nn.Sequential(
            nn.Conv2d(5, 3, kernel_size=1, bias=False)
        )
        self.vit.head = torch.nn.Identity() #remove last linear layer

        if config.vit.pretrained:
            checkpoint = torch.load(config.vit.pretrained_weights, map_location="cpu")
            vo_regressor_state_dict = checkpoint['model_state_dict']

            vit_state_dict = OrderedDict()
            adapter_state_dict = OrderedDict()

            for key, value in vo_regressor_state_dict.items():
                if key.startswith('encoder.vit.'):
                    new_key = key.replace('encoder.vit.', '', 1)
                    vit_state_dict[new_key] = value
                elif key.startswith('adapter.'):
                    new_key = key.replace('adapter.', '', 1)
                    adapter_state_dict[new_key] = value

            # Load the filtered weights
            self.vit.load_state_dict(vit_state_dict, strict=True)
            self.adapter.load_state_dict(adapter_state_dict, strict=True)


        
    def forward(self, data):
        im0 = torch.clamp(data['view0']['image'], 0, 1)
        im1 = torch.clamp(data['view1']['image'], 0, 1)

        intrinsics_layer = make_intrinsics_layer(
            im0.shape[2], im0.shape[3], data['K']
        )
        im0 = torch.cat([im0, intrinsics_layer], dim=1) # B x 5 x H x W
        im1 = torch.cat([im1, intrinsics_layer], dim=1) # B x 5 x H x W

        im0 = self.adapter(im0) # B x 3 x H x W
        im1 = self.adapter(im1) # B x 3 x H x W

        im0 = im0.unsqueeze(1)
        im1 = im1.unsqueeze(1)
        x = torch.cat([im0, im1], dim=1)

        # Reshape for ViT (B x C x T x H x W)
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

class CNNMotionEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim_emb = config.photo_vo.model.dim_emb
        
        self.feature_map_size = (192, 128)  # (H_feat, W_feat)
        
        self.global_local_fusion = nn.Sequential(
            nn.Linear(self.dim_emb * 2, self.dim_emb),  
            nn.ReLU(),
            nn.LayerNorm(self.dim_emb)
        )
        
        self.feature_densifier = nn.Sequential(
            nn.Conv2d(self.dim_emb, self.dim_emb, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.dim_emb, self.dim_emb, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.projection = nn.Conv2d(self.dim_emb, 4, kernel_size=1)
        
        self.regressor = VOFlowRes()
        self.load_pretrained("weights/tartanvo_1914.pkl")

        self.original_img_height = config.vit.image_size[0]
        self.original_img_width = config.vit.image_size[1]

    def load_pretrained(self, path):
        print(f"Loading pretrained VOFlowRes weights from {path}...")
        voflow_state_dict = torch.load(path, map_location="cpu")
        
        renamed_state_dict = OrderedDict()
        prefix_to_find = 'flowPoseNet.'
        
        for key, value in voflow_state_dict.items():
            if key.startswith(prefix_to_find):
                new_key = key.replace(prefix_to_find, 'regressor.', 1)
                renamed_state_dict[new_key] = value

        if not renamed_state_dict:
            print("No keys matched 'flowPoseNet.'. Trying 'module.flowPoseNet.' for DataParallel model...")
            prefix_to_find = 'module.flowPoseNet.'
            for key, value in voflow_state_dict.items():
                if key.startswith(prefix_to_find):
                    new_key = key.replace(prefix_to_find, 'regressor.', 1)
                    renamed_state_dict[new_key] = value

        if not renamed_state_dict:
            print("Warning: Could not find any VOFlowRes weights in the checkpoint.")
            return

        self.load_state_dict(renamed_state_dict, strict=False)

    def forward(self, image_embs, patch_embs, patch_coords):
        B, N, D = patch_embs.shape
        H_feat, W_feat = self.feature_map_size
        
        # Fuse global and local features
        global_expanded = image_embs.unsqueeze(1).expand(-1, N, -1)
        fused_features = torch.cat([patch_embs, global_expanded], dim=-1)
        fused_features = self.global_local_fusion(fused_features)

        # Create initial sparse feature map
        feature_map = torch.zeros(B, D, H_feat, W_feat, device=patch_embs.device, dtype=patch_embs.dtype)
        weight_map = torch.zeros(B, 1, H_feat, W_feat, device=patch_embs.device, dtype=patch_embs.dtype)
        
        # Scale coordinates
        scaled_coords_x = torch.clamp(
            (patch_coords[..., 0] / self.original_img_width * (W_feat - 1)),
            min=0, max=W_feat - 1
        )
        scaled_coords_y = torch.clamp(
            (patch_coords[..., 1] / self.original_img_height * (H_feat - 1)),
            min=0, max=H_feat - 1
        )
        
        # Handle invalid coordinates (e.g., -1 for non-matches)
        valid_mask = (patch_coords[..., 0] >= 0)
        
        # Use bilinear interpolation to place features more smoothly
        for b in range(B):
            valid_indices = valid_mask[b]
            if not valid_indices.any():
                continue
                
            valid_x = scaled_coords_x[b][valid_indices]
            valid_y = scaled_coords_y[b][valid_indices]
            valid_feats = fused_features[b][valid_indices]
            
            x0 = torch.floor(valid_x).long()
            y0 = torch.floor(valid_y).long()
            x1 = torch.clamp(x0 + 1, max=W_feat - 1)
            y1 = torch.clamp(y0 + 1, max=H_feat - 1)
            
            wx = valid_x - x0.float()
            wy = valid_y - y0.float()
            
            w00 = ((1 - wx) * (1 - wy)).unsqueeze(1)
            w01 = ((1 - wx) * wy).unsqueeze(1)
            w10 = (wx * (1 - wy)).unsqueeze(1)
            w11 = (wx * wy).unsqueeze(1)

            fm_2d_t = torch.zeros(H_feat * W_feat, D, device=patch_embs.device, dtype=patch_embs.dtype)
            wm_1d = weight_map[b, 0].view(H_feat * W_feat)
            
            # Calculate 1D indices for each of the 4 corners
            idx00 = y0 * W_feat + x0
            idx10 = y1 * W_feat + x0
            idx01 = y0 * W_feat + x1
            idx11 = y1 * W_feat + x1

            fm_2d_t.index_add_(0, idx00, w00 * valid_feats)
            fm_2d_t.index_add_(0, idx10, w01 * valid_feats)
            fm_2d_t.index_add_(0, idx01, w10 * valid_feats)
            fm_2d_t.index_add_(0, idx11, w11 * valid_feats)
            feature_map[b] = fm_2d_t.T.view(D, H_feat, W_feat)

            wm_1d = torch.zeros(H_feat * W_feat, device=patch_embs.device, dtype=patch_embs.dtype)
            wm_1d.index_add_(0, idx00, w00.squeeze(1))
            wm_1d.index_add_(0, idx10, w01.squeeze(1))
            wm_1d.index_add_(0, idx01, w10.squeeze(1))
            wm_1d.index_add_(0, idx11, w11.squeeze(1))
            weight_map[b, 0] = wm_1d.view(H_feat, W_feat)
            
        # Reshape feature map back to 2D
        feature_map = feature_map.view(B, D, H_feat, W_feat)
        weight_map = weight_map.view(B, 1, H_feat, W_feat)
        
        # Normalize by weights to handle overlapping features
        weight_map = torch.clamp(weight_map, min=1e-8)
        feature_map = feature_map / weight_map
        
        # Densify the sparse feature map
        dense_features = self.feature_densifier(feature_map)
        
        # Project to 4 channels for VOFlowRes
        projected_map = self.projection(dense_features)
        pose = self.regressor(projected_map)
        return pose
    

class PhotoVoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.imgenc = ImagePairEncoder(config)
        self.matcher = gf.models.get_model(config.features_model.name)(config.features_model)
        self.penc = PatchEncoder(config)
        self.motion_estimator = CNNMotionEstimator(config)
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


        # make intrinsics layer
        il = make_intrinsics_layer(data['view0']['image'].shape[2],
                                   data['view0']['image'].shape[3],
                                   data['K'])
        # Add intrinsics layer to images
        im0 = torch.cat([data['view0']['image'], il], dim=1)
        im1 = torch.cat([data['view1']['image'], il], dim=1)

        # Extract local patches from both views using valid keypoints
        patches0 = get_patches(im0, kpts0_valid, self.config.photo_vo.model.patch_size)
        patches1 = get_patches(im1, kpts1_valid, self.config.photo_vo.model.patch_size)

        # Store patch-related info in data
        data['view0']['patches'] = patches0
        data['view1']['patches'] = patches1
        data['view0']['patches_coords'] = kpts0_valid
        data['view1']['patches_coords'] = kpts1_valid
        data['view0']['scores'] = scores0_valid
        data['view1']['scores'] = scores1_valid

        patch_embs = self.penc(data) 
        output = self.motion_estimator(image_embs, patch_embs, torch.cat([kpts0_valid, kpts1_valid], dim=1))
        data['pred_vo'] = output

        return {**data, **self.features}

def get_photo_vo_model(config):
    return PhotoVoModel(config)