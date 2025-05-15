import glob
import os
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import random


def rotation_to_euler(M, cy_thresh=None, seq='zyx'):
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = np.finfo(float).eps * 4.0  # _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = np.sqrt(r33 * r33 + r23 * r23)
    if seq == 'zyx':
        if cy > cy_thresh:  # cos(y) not close to zero, standard form
            z = np.arctan2(-r12, r11)  # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = np.arctan2(r13, cy)  # atan2(sin(y), cy)
            x = np.arctan2(-r23, r33)  # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else:  # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = np.arctan2(r21, r22)
            y = np.arctan2(r13, cy)  # atan2(sin(y), cy)
            x = 0.0
    elif seq == 'xyz':
        if cy > cy_thresh:
            y = np.arctan2(-r31, cy)
            x = np.arctan2(r32, r33)
            z = np.arctan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = np.pi / 2
                x = np.arctan2(r12, r13)
            else:
                y = -np.pi / 2
    else:
        raise Exception('Sequence not recognized')
    return [z, y, x]


class KITTI(torch.utils.data.Dataset):
    def __init__(self,
                 data_path=r"data/sequences_jpg",
                 gt_path=r"data/poses",
                 camera_id="2",
                 sequences=["00", "02", "08", "09"],
                 apply_rcr=False,
                 max_skip=0,
                 transform=None,
                 resize=(640, 640)):
        self.data_path = data_path
        self.gt_path = gt_path
        self.camera_id = camera_id
        self.max_skip = max_skip # max number of frames to skip
        self.transform = transform
        self.apply_rcr = apply_rcr
        self.resize = resize
        # KITTI normalization
        # self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        # self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        # self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        # self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        self.sequences = sequences
        
        frames, seqs = self.read_frames()
        gt = self.read_gt()
        Ks = self.read_K()
        self.pairs = self.create_pairs(frames, seqs, gt, Ks)


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
                pose1 = np.array(gt[idx]).reshape(3,4)
                pose2 = np.array(gt[next_idx]).reshape(3,4)

                pairs.append({
                    'frame1': frame,
                    'frame2': frames[next_idx],
                    'pose1': pose1,
                    'pose2': pose2,
                    'K': Ks[seq]
                })
        
        return pairs

    def compute_relative_pose(self, pose1, pose2):
        # convert poses to homogeneous coordinates
        T1 = np.vstack([pose1, [0, 0, 0, 1]])
        T2 = np.vstack([pose2, [0, 0, 0, 1]])
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
    
    def read_frames(self):
        # Get frames list
        frames = []
        seqs = []
        for sequence in self.sequences:
            frames_dir = os.path.join(self.data_path, sequence, "image_{}".format(self.camera_id), "*.png")
            frames_seq = sorted(glob.glob(frames_dir))
            frames = frames + frames_seq
            seqs = seqs + [sequence] * len(frames_seq)
        return frames, seqs

    def read_gt(self):
        # Read ground truth
        gt = []
        for sequence in self.sequences:
            with open(os.path.join(self.gt_path, sequence + ".txt")) as f:
                lines = f.readlines()

            # convert poses to float
            for line_idx, line in enumerate(lines):
                line = line.strip().split()
                line = [float(x) for x in line]
                gt.append(line)
        return gt
    
    def read_K(self):
        # Read camera intrinsics
        K_dict = {}
        for i, sequence in enumerate(self.sequences):
            calib_path = os.path.join(self.data_path, sequence, 'calib.txt')
            with open(calib_path) as f:
                lines = f.readlines()
            # Parse the calibration line for camera_id
            cam_line = [line for line in lines if line.startswith(f'P{self.camera_id}:')][0]
            parts = cam_line.strip().split()
            fx, fy, cx, cy = float(parts[1]), float(parts[6]), float(parts[3]), float(parts[7])
            K_dict[sequence] = [fx, fy, cx, cy]
        return K_dict


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

    
        
