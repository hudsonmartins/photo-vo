import glob
import os
import numpy as np
from PIL import Image
import torch
import random
from scipy.spatial.transform import Rotation as R


def rotation_to_euler(R_matrix, seq='zyx'):
    return R.from_matrix(R_matrix).as_euler(seq)

class KITTI(torch.utils.data.Dataset):
    def __init__(self,
                 data_path=r"data/sequences",
                 gt_path=r"data/poses",
                 camera_id="2",
                 sequences=["00", "02", "08", "09"],
                 apply_rcr=False,
                 max_skip=0,
                 transform=None,
                 resize=(640, 640)):
        # normalization
        self.mean_angles = np.array([0.0, 0.0, 0.0])
        self.std_angles = np.array([0.01,0.01,0.01])
        self.mean_t = np.array([0.0, 0.0, 0.0])
        self.std_t = np.array([0.2,0.2,0.2])
        
        self.data_path = data_path
        self.gt_path = gt_path
        self.camera_id = camera_id
        self.max_skip = max_skip # max number of frames to skip
        self.transform = transform
        self.apply_rcr = apply_rcr
        self.resize = resize

        self.sequences = sequences
        
        self.sequences_data = self.read_sequences()
        self.pairs = self.create_pairs()


    def read_sequences(self):
        sequences_data = {}
        for seq in self.sequences:
            # Read image paths
            image_dir = os.path.join(self.data_path, seq, f'image_{self.camera_id}')
            images = sorted(glob.glob(os.path.join(image_dir, '*.png')))

            # Read intrinsics
            calib_file = os.path.join(self.data_path, seq, 'calib.txt')
            with open(calib_file) as f:
                lines = f.readlines()
            cam_line = [line for line in lines if line.startswith(f'P{self.camera_id}:')][0]
            parts = cam_line.strip().split()
            fx, fy, cx, cy = float(parts[1]), float(parts[6]), float(parts[3]), float(parts[7])
            K = [fx, fy, cx, cy]

            # Read poses
            with open(os.path.join(self.gt_path, seq + ".txt")) as f:
                lines = f.readlines()
            poses = [list(map(float, line.strip().split())) for line in lines]

            sequences_data[seq] = {
                'images': images,
                'poses': poses,
                'K': K
            }

        return sequences_data

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

                pose1 = np.array(poses[i]).reshape(3, 4)
                pose2 = np.array(poses[j]).reshape(3, 4)

                pairs.append({
                    'frame1': images[i],
                    'frame2': images[j],
                    'pose1': pose1,
                    'pose2': pose2,
                    'K': K
                })
        return pairs

    def compute_relative_pose(self, pose1, pose2):
        T1 = np.vstack([pose1, [0, 0, 0, 1]])
        T2 = np.vstack([pose2, [0, 0, 0, 1]])
        T_rel = np.dot(np.linalg.inv(T1), T2)

        R = T_rel[:3, :3]
        t = T_rel[:3, 3]
        angles = rotation_to_euler(R, seq='zyx')
        angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
        t = (np.asarray(t) - self.mean_t) / self.std_t
        angles = np.nan_to_num(angles, 0.0)
        t = np.nan_to_num(t, 0.0)

        return torch.FloatTensor(np.concatenate([angles, t]))

    def rcr(self, img1, img2, K):
        original_size = img1.size
        crop_scale = random.uniform(0.4, 1.0)
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
            original_size = img1.size
            img1 = img1.resize(self.resize)
            img2 = img2.resize(self.resize)
            fx, fy, cx, cy = pair['K']
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