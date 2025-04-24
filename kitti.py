import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import random


def add_gamma(img):
    gamma = random.uniform(0.5, 2.0)
    return img ** gamma

def add_occlusion(img):
    h, w = img.shape[1:]
    mask = torch.ones_like(img)

    occ_h = h // 4  # height of occlusion
    occ_w = w // 4  # width of occlusion

    center_h = h // 2
    center_w = w // 2

    mask[:, center_h - occ_h//2 : center_h + occ_h//2,
             center_w - occ_w//2 : center_w + occ_w//2] = 0
    return img * mask


def get_iterator(data_path, size, batch_size, sequences_names, max_skip, train=True):
    if(train):
        preprocess = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.0))], p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: add_gamma(x) if random.random() < 1.0 else x),            
            transforms.Lambda(lambda x: add_occlusion(x) if random.random() < 0.2 else x)
        ])
        
        kitti = KITTI(os.path.join(data_path, 'sequences'), 
                    os.path.join(data_path, 'poses'), 
                    transform=preprocess, 
                    sequences=sequences_names, 
                    max_skip=max_skip
                )
        return torch.utils.data.DataLoader(kitti, batch_size=batch_size, shuffle=True)
    else:
        preprocess = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        
        kitti = KITTI(os.path.join(data_path, 'sequences'), 
                    os.path.join(data_path, 'poses'), 
                    transform=preprocess, 
                    sequences=sequences_names, 
                    max_skip=max_skip
                )
        return torch.utils.data.DataLoader(kitti, batch_size=batch_size, shuffle=False)
    

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
                 max_skip=0,
                 transform=None,
                 ):

        self.data_path = data_path
        self.gt_path = gt_path
        self.camera_id = camera_id
        self.max_skip = max_skip # max number of frames to skip
        self.transform = transform

        # KITTI normalization
        self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        self.sequences = sequences

        frames, seqs = self.read_frames()
        gt = self.read_gt()

        self.pairs = self.create_pairs(frames, seqs, gt)

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
        if self.read_gt:
            gt = []
            for sequence in self.sequences:
                with open(os.path.join(self.gt_path, sequence + ".txt")) as f:
                    lines = f.readlines()

                # convert poses to float
                for line_idx, line in enumerate(lines):
                    line = line.strip().split()
                    line = [float(x) for x in line]
                    gt.append(line)

        else:
            gt = None

        return gt
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # load images
        img1 = Image.open(pair['frame1']).convert('RGB')
        img2 = Image.open(pair['frame2']).convert('RGB')
           
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
        return torch.FloatTensor(imgs), rel_pose

    def create_pairs(self, frames, seqs, gt):
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
                    'pose2': pose2
                })
        
        return pairs

    def compute_relative_pose(self, pose1, pose2):
        # convert poses to homogeneous coordinates
        T1 = np.vstack([pose1, [0, 0, 0, 1]])
        T2 = np.vstack([pose2, [0, 0, 0, 1]])
        
        T_rel = np.linalg.inv(T1) @ T2
        
        R = T_rel[:3, :3]
        t = T_rel[:3, 3]
        
        angles = rotation_to_euler(R, seq='zyx')
        angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
        t = (np.asarray(t) - self.mean_t) / self.std_t
        
        return torch.FloatTensor(np.concatenate([angles, t]))

if __name__ == "__main__":
    # Example usage    
    loader = get_iterator(
        data_path='/home/hudson/Desktop/Unicamp/datasets/kitti',
        size=(640, 640),
        batch_size=5,
        sequences_names=["03"],
        max_skip=0
    )
    for i, (imgs, y) in enumerate(loader):
        print(imgs)
        print(y)
        print(imgs.shape)
        print(y.shape)
        
        imgs = imgs

        img1 = imgs[0][0].permute(1, 2, 0).numpy()
        img2 = imgs[0][1].permute(1, 2, 0).numpy()
        img1 = (img1 * 255).astype(np.uint8)
        img2 = (img2 * 255).astype(np.uint8)
        
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.show()
        plt.close()