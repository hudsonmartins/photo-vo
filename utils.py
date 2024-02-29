import torch
import cv2
import numpy as np


def get_torch_not_nan(tensor):
    """Returns a tensor without nan values"""
    return tensor[~torch.isnan(tensor)]

    
def draw_patches(img, kpts, color=(0,0,255), patch_size=10):
    """Draw patches around keypoints on an image"""
    half = patch_size // 2
    for pt in kpts:
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        img = cv2.rectangle(
            np.ascontiguousarray(img).astype(np.uint8),
            (int(pt[0]) - half, int(pt[1]) - half),
            (int(pt[0]) + half, int(pt[1]) + half),
            color,
            2,
        )
    return img


def draw_pts(img, pts, color=(0, 0, 255), radius=5):
    """Draw points on an image"""
    for pt in pts:
        #check if pt is nan
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        img = cv2.circle(np.ascontiguousarray(img).astype(np.uint8), (int(pt[0]), int(pt[1])), radius, color, 2)
    return img



def get_patches(img, pts, patch_size=10):
    """Given an image and a set of points, return the patches around the points"""

    batch_size = img.size(0)
    patch_size = int(patch_size)
    half = patch_size // 2
    patches = []
    for i in range(batch_size):
        #pad image
        img_pad = torch.nn.functional.pad(img[i], (half, half, half, half), mode='reflect')
        pts_i = pts[i]
        patches_i = []
        for j in range(pts_i.size(0)):
            if torch.isnan(pts_i[j]).any():
                patches_i.append(torch.zeros(3, patch_size, patch_size))
                continue
            x, y = pts_i[j].int()            
            patch = img_pad[..., y:y+patch_size, x:x+patch_size]
            patches_i.append(patch)
        patches.append(torch.stack(patches_i))

    return torch.stack(patches)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread("/home/hudson/Desktop/Unicamp/datasets/megadepth1500/images/0015/29307281_d7872975e2_o.jpg")
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255
    img = img.unsqueeze(0)
    kpts = torch.tensor([[100, 10], [20, 200], [1200, 900], [500, 500]]).unsqueeze(0).float()
    print('img size ', img.size())
    print('kpts shape ', kpts.size())   
    patches = get_patches(img, kpts)
    
    img = img.squeeze(0).permute(1, 2, 0).numpy()
    kpts = kpts.squeeze(0).numpy().astype(int)
    for pt in kpts:
        img = cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), 2)
    
    print('patches ', patches.size())
