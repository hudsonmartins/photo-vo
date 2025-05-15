import os
import glob
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def rotation_to_euler(R_matrix, seq='zyx'):
    return R.from_matrix(R_matrix).as_euler(seq)

class QueensCAMP(Dataset):
    def __init__(self,
                 data_path,
                 sequences,
                 apply_rcr=False,
                 resize=(640, 640),
                 max_skip=0,
                 transform=None):

        self.data_path = data_path
        self.max_skip = max_skip
        self.transform = transform
        self.resize = resize
        self.apply_rcr = apply_rcr
        # KITTI normalization
        #self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        #self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        #self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        #self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        self.sequences = sequences

        frames, seqs = self.read_frames()
        gt = self.read_gt()
        Ks = self.read_K()
        self.pairs = self.create_pairs(frames, seqs, gt, Ks)
        

    def read_frames(self):
        # Get frames list
        frames = []
        seqs = []
        for sequence in self.sequences:
            frames_dir = os.path.join(self.data_path, 
                                      "sequences", 
                                      sequence, 
                                      "images", 
                                      "*.png")
            frames_seq = sorted(glob.glob(frames_dir))
            frames = frames + frames_seq
            seqs = seqs + [sequence] * len(frames_seq)
        return frames, seqs


    def read_gt(self):
        # Read ground truth
        gt = []
        for sequence in self.sequences:
            gt_path = os.path.join(self.data_path,
                                    "sequences", 
                                    sequence, 
                                    "traj.txt")
            with open(os.path.join(gt_path)) as f:
                for line in f:
                    tokens = line.strip().split()
                    tx, ty, tz = map(float, tokens[1:4])
                    qx, qy, qz, qw = map(float, tokens[4:8])
                    R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    T = np.eye(4)
                    T[:3, :3] = R_mat
                    T[:3, 3] = [tx, ty, tz]
                    gt.append(T)
        return gt
    
    def read_K(self):
        calib_path = os.path.join(self.data_path, 'rgb_camera_info.txt')
        with open(calib_path, 'r') as f:
            params = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key in ['height', 'width']:
                    params[key] = int(value)
                elif key == 'distortion_model':
                    params[key] = value.strip('"')
                else:
                    value = value.strip('[]')
                    value_list = [float(v.strip()) for v in value.split(',')]
                    params[key] = value_list

        # Extract fx, fy, cx, cy from K
        fx = params['K'][0]
        fy = params['K'][4]
        cx = params['K'][2]
        cy = params['K'][5]

        K_dict = {}
        for sequence in self.sequences:
            K_dict[sequence] = [fx, fy, cx, cy]
        return K_dict

    def convert_to_kitti_frame(self, pose):
        # KITTI: x (right), y (down), z (forward)
        # QueensCAMP: x (forward), y (left), z (up)
        R_AB = np.array([
            [0, -1,  0],
            [0,  0, -1],
            [1,  0,  0]
        ], dtype=np.float32)
        R_q = pose[:3, :3]
        t_q = pose[:3, 3]
        R_k = R_AB @ R_q
        t_k = R_AB @ t_q.reshape(-1, 1)
        pose_k = np.eye(4, dtype=np.float32)
        pose_k[:3, :3] = R_k
        pose_k[:3, 3] = t_k.squeeze()
        return pose_k

    def compute_relative_pose(self, pose1, pose2):
        # convert poses to homogeneous coordinates
        T1 = self.convert_to_kitti_frame(pose1)
        T2 = self.convert_to_kitti_frame(pose2)

        # compute relative pose
        T_rel = np.dot(np.linalg.inv(T1), T2)
        
        R = T_rel[:3, :3]
        t = T_rel[:3, 3]
        
        # Normalize angles and translation
        angles = rotation_to_euler(R, seq='zyx')
        #angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
        #t = (np.asarray(t) - self.mean_t) / self.std_t
        
        angles = np.nan_to_num(angles, 0.0)
        t = np.nan_to_num(t, 0.0)
        
        return torch.FloatTensor(np.concatenate([angles, t]))


    def create_pairs(self, frames, seqs, gt, Ks):
        pairs = []
        current_seq = None
        
        for idx, (frame, seq) in enumerate(zip(frames, seqs)):
            if seq != current_seq:
                current_seq = seq
                
            skip = random.randint(1, self.max_skip+1)
            next_idx = idx + skip
            
            # check if next frame is in the same sequence and within bounds
            if next_idx < len(frames) and seqs[next_idx] == seq:
                pose1 = np.array(gt[idx])
                pose2 = np.array(gt[next_idx])
                
                pairs.append({
                    'frame1': frame,
                    'frame2': frames[next_idx],
                    'pose1': pose1,
                    'pose2': pose2,
                    'K': Ks[seq]
                })
        
        return pairs


    def rcr(self, img1, img2, K):
        # Apply RCR
        original_size = img1.size
        
        # Random crop parameters
        crop_scale = random.uniform(0.4, 1.0)
        crop_width = int(original_size[0] * crop_scale)
        crop_height = int(original_size[1] * crop_scale)
        x0 = random.randint(0, original_size[0] - crop_width)
        y0 = random.randint(0, original_size[1] - crop_height)

        # Crop and resize
        img1 = img1.crop((x0, y0, x0 + crop_width, y0 + crop_height)).resize(self.resize)
        img2 = img2.crop((x0, y0, x0 + crop_width, y0 + crop_height)).resize(self.resize)

        fx, fy, cx, cy = K
        # Adjust intrinsics
        sx = self.resize[0] / crop_width
        sy = self.resize[1] / crop_height
        fx_new = fx * sx
        fy_new = fy * sy
        cx_new = (cx - x0) * sx
        cy_new = (cy - y0) * sy
        return img1, img2, [fx_new, fy_new, cx_new, cy_new]
    

    def __len__(self):
        return len(self.pairs)
    

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # load images
        img1 = Image.open(pair['frame1']).convert('RGB')
        img2 = Image.open(pair['frame2']).convert('RGB')
        
        if self.apply_rcr:
            img1, img2, K = self.rcr(img1, img2, pair['K'])
        else:
            K = pair['K']
            original_size = img1.size
            img1 = img1.resize(self.resize)
            img2 = img2.resize(self.resize)
            
            fx, fy, cx, cy = K
            sx = self.resize[0] / original_size[0]
            sy = self.resize[1] / original_size[1]
            K = [fx * sx, fy * sy, cx * sx, cy * sy]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        # load poses
        pose1 = pair['pose1']
        pose2 = pair['pose2']
        # compute relative pose
        rel_pose = self.compute_relative_pose(pose1, pose2)
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        imgs = np.concatenate([img1, img2], axis=0)
        imgs = np.asarray(imgs)
        return torch.FloatTensor(imgs), rel_pose, torch.FloatTensor(K)