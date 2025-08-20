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
        # normalization
        self.mean_angles = np.array([0.0, 0.0, 0.0])
        self.std_angles = np.array([0.01,0.01,0.01])
        self.mean_t = np.array([0.0, 0.0, 0.0])
        self.std_t = np.array([0.2,0.2,0.2])

        self.data_path = data_path
        self.max_skip = max_skip
        self.transform = transform
        self.resize = resize
        self.apply_rcr = apply_rcr

        self.sequences = sequences

        self.sequences_data = self.read_sequences()
        self.pairs = self.create_pairs()
        

    def read_sequences(self):
        sequences_data = {}
        for seq in self.sequences:
            # Read image paths
            image_dir = os.path.join(self.data_path, "sequences", seq, "images")
            images = sorted(glob.glob(os.path.join(image_dir, '*.png')))

            # Read intrinsics
            calib_file = os.path.join(self.data_path, 'rgb_camera_info.txt')
            with open(calib_file, 'r') as f:
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

                fx = params['K'][0]
                fy = params['K'][4]
                cx = params['K'][2]
                cy = params['K'][5]
                K = [fx, fy, cx, cy]

            # Read poses
            with open(os.path.join(self.data_path, "sequences", seq, "traj.txt")) as f:
                lines = f.readlines()
            poses = []
            for line in lines:
                tokens = line.strip().split()
                tx, ty, tz = map(float, tokens[1:4])
                qx, qy, qz, qw = map(float, tokens[4:8])
                R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = R_mat
                T[:3, 3] = [tx, ty, tz]
                poses.append(T)
            poses = np.array(poses, dtype=np.float32)

            sequences_data[seq] = {
                'images': images,
                'poses': poses,
                'K': K
            }

        return sequences_data
    
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
        angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
        t = (np.asarray(t) - self.mean_t) / self.std_t
        
        angles = np.nan_to_num(angles, 0.0)
        t = np.nan_to_num(t, 0.0)
        
        return torch.FloatTensor(np.concatenate([angles, t]))


    def create_pairs(self):
        pairs = []
        for seq, data in self.sequences_data.items():
            images = data['images']
            poses = data['poses']
            K = data['K']

            for i in range(len(images) - 1):
                skip = random.randint(1, self.max_skip + 1)
                j = i + skip
                if j >= len(images):
                    continue

                pose1 = np.array(poses[i])
                pose2 = np.array(poses[j])

                pairs.append({
                    'frame1': images[i],
                    'frame2': images[j],
                    'pose1': pose1,
                    'pose2': pose2,
                    'K': K
                })
        return pairs


    def rcr(self, img1, img2, K):
        # Apply RCR
        original_size = img1.size
        
        # Random crop parameters
        crop_scale = random.uniform(0.7, 1.0)
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
        
        img1_path, img2_path = pair['frame1'], pair['frame2']
        pose1, pose2 = pair['pose1'], pair['pose2']
        # Randomly swap images and poses for data augmentation
        if self.transform and np.random.rand() < 0.5:
            img1_path, img2_path = img2_path, img1_path
            pose1, pose2 = pose2, pose1

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
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
            
                
        rel_pose = self.compute_relative_pose(pose1, pose2)

        # Combine into tensor
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        imgs = torch.cat([img1, img2], dim=0)

        return imgs, rel_pose, torch.FloatTensor(K)