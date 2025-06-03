import os
import torch
import random
from PIL import Image
from torchvision import transforms
from queenscamp import QueensCAMP
from kitti import KITTI

def add_gamma(img):
    # Apply gamma correction
    gamma = random.uniform(0.75, 1.25)
    img = torch.clamp(img, min=1e-5)
    return img ** gamma


def add_occlusion(img):
    # Get image dimensions (C, H, W)
    c, h, w = img.shape
    
    # Create mask with same shape as image
    mask = torch.ones_like(img)
    
    # Occlusion size (1/4 of image dimensions)
    occ_h = h // 4
    occ_w = w // 4
    
    # Random starting positions (ensure occlusion stays within image bounds)
    y_start = torch.randint(0, h - occ_h, (1,)).item()
    x_start = torch.randint(0, w - occ_w, (1,)).item()
    
    # Calculate end positions
    y_end = y_start + occ_h
    x_end = x_start + occ_w
    
    # Apply occlusion mask
    mask[:, y_start:y_end, x_start:x_end] = 0
    
    return img * mask


def get_iterator(datasets_names, train, **kwargs):
    """
    Create a combined iterator for multiple datasets.
    """
    datasets = []
    for dataset in datasets_names:
        if(dataset == 'kitti'):
            datasets.append(get_kitti(**kwargs['kitti'], train=train))
        elif(dataset == 'queenscamp'):
            datasets.append(get_queenscamp(**kwargs['queenscamp'], train=train))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
    return torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(datasets), batch_size=kwargs['batch_size'], shuffle=train)
   

def get_kitti(data_path, size, train_sequences, val_sequences, max_skip, train):
    if(train):
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: add_gamma(x) if random.random() < 0.3 else x),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.0))], p=0.3),
            transforms.Lambda(lambda x: add_occlusion(x) if random.random() < 0.1 else x)
        ])
    else:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
    return KITTI(os.path.join(data_path, 'sequences'), 
                os.path.join(data_path, 'poses'), 
                transform=preprocess, 
                sequences=train_sequences if train else val_sequences,
                resize=size,
                apply_rcr=train,
                max_skip=max_skip)
    

def get_queenscamp(data_path, size, train_sequences, val_sequences, max_skip, train):
    if train:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.0))], p=0.3),
            transforms.Lambda(lambda x: add_occlusion(x) if random.random() < 0.1 else x)
        ])
    else:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

    return QueensCAMP(data_path=data_path,
                      resize=size,
                      apply_rcr=train,
                      max_skip=max_skip,
                      sequences=train_sequences if train else val_sequences, 
                      transform=preprocess)
    
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    loader = get_iterator(
        datasets_names=['kitti', 'queenscamp'],
        train=True,
        kitti = {
            'data_path': '/home/hudson/Desktop/Unicamp/datasets/kitti',
            'size': (640, 640),
            'train_sequences': ["01","08"],
            'val_sequences': ["08"],
            'max_skip': 5
        },
        queenscamp = {
            'data_path': '/home/hudson/Desktop/Unicamp/datasets/queenscamp',
            'train_sequences': ["16"],
            'val_sequences': ["16"],
            'size': (640, 640),
            'max_skip': 5
        },
        batch_size=1
    )

    for i, (imgs, y, K) in enumerate(loader):  
        print('y ', y)
        print('K ', K)
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
        