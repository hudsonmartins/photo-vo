import os
import glob
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R


def rotation_to_euler(R_matrix, seq='zyx'):
    return R.from_matrix(R_matrix).as_euler(seq)


class TartanAir(Dataset):
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
        self.sequences = sequences #in TartanAir we treat a sequence as a path from root folder environment/difficulty/trajectory

        self.data = self.load_sequences()
        self.pairs = self.create_pairs()

    def load_sequences(self):
        data = {}
        intrinsics = self.read_K()
        for sequence in self.sequences:
            frame_paths = sorted(glob.glob(os.path.join(self.data_path, sequence, "image_left", "*.png")))

            # Load poses
            pose_file = os.path.join(self.data_path, sequence, "pose_left.txt")
            poses = []
            with open(pose_file, 'r') as f:
                for line in f:
                    tokens = line.strip().split()
                    tx, ty, tz = map(float, tokens[0:3])
                    qx, qy, qz, qw = map(float, tokens[3:7])
                    R_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    T = np.eye(4)
                    T[:3, :3] = R_mat
                    T[:3, 3] = [tx, ty, tz]
                    poses.append(T)

            data[sequence] = {
                'frames': frame_paths,
                'poses': poses,
                'K': intrinsics[sequence]
            }
        return data

    def read_K(self):
        calib_path = os.path.join(self.data_path, 'rgb_camera_info.txt')
        with open(calib_path, 'r') as f:
            params = {}
            for line in f:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().split('#')[0].strip()
                if key in ['width', 'height']:
                    params[key] = int(value)
                elif key in ['fx', 'fy', 'cx', 'cy']:
                    params[key] = float(value)

        fx = params['fx']
        fy = params['fy']
        cx = params['cx']
        cy = params['cy']

        return {sequence: [fx, fy, cx, cy] for sequence in self.sequences}

    def convert_to_kitti_frame(self, pose):
        # KITTI: x (right), y (down), z (forward)
        # TartanAir: x (forward), y (right), z (down)
        R_AB = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
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
        T1 = self.convert_to_kitti_frame(pose1)
        T2 = self.convert_to_kitti_frame(pose2)
        # convert poses to homogeneous coordinates
        T_rel = np.dot(np.linalg.inv(T1), T2)
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]
        
        angles = rotation_to_euler(R_rel, seq='zyx')
        # Normalize angles and translation
        angles = (angles - self.mean_angles) / self.std_angles 
        t = (t_rel - self.mean_t) / self.std_t
        angles = np.nan_to_num(angles, 0.0)
        t = np.nan_to_num(t, 0.0)
        return torch.FloatTensor(np.concatenate([angles, t]))

    def create_pairs(self):
        '''
            Create pairs of frames with random skips.
        '''
        pairs = []
        for seq, seq_data in self.data.items():
            frames = seq_data['frames']
            poses = seq_data['poses']
            K = seq_data['K']

            for idx in range(len(frames) - 1):
                skip = random.randint(1, self.max_skip + 1)
                next_idx = idx + skip
                if next_idx >= len(frames):
                    continue

                pair = {
                    'frame1': frames[idx],
                    'frame2': frames[next_idx],
                    'pose1': poses[idx],
                    'pose2': poses[next_idx],
                    'K': K
                }
                pairs.append(pair)
        return pairs

    def rcr(self, img1, img2, K):
        '''
            Randomly crops a region from the images and adjusts the camera intrinsics accordingly.
        '''
        original_size = img1.size
        crop_scale = random.uniform(0.7, 1.0)
        crop_width = int(original_size[0] * crop_scale)
        crop_height = int(original_size[1] * crop_scale)
        x0 = random.randint(0, original_size[0] - crop_width)
        y0 = random.randint(0, original_size[1] - crop_height)

        img1 = img1.crop((x0, y0, x0 + crop_width, y0 + crop_height)).resize(self.resize)
        img2 = img2.crop((x0, y0, x0 + crop_width, y0 + crop_height)).resize(self.resize)

        fx, fy, cx, cy = K
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

        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        imgs = torch.cat([img1, img2], dim=0)

        return imgs, rel_pose, torch.FloatTensor(K)
