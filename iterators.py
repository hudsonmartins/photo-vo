import os
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    balanced_sampling = kwargs.get('balanced_sampling', True) if train else False
    for dataset in datasets_names:
        if(dataset == 'kitti'):
            datasets.append(get_kitti(**kwargs['kitti'], train=train))
        elif(dataset == 'queenscamp'):
            datasets.append(get_queenscamp(**kwargs['queenscamp'], train=train))
        elif(dataset == 'tartanair'):
            datasets.append(get_tartanair(**kwargs['tartanair'], train=train))
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
    concat = ConcatDataset(datasets)

    # Balanced sampling across datasets using per-sample weights (optional)
    if balanced_sampling and len(datasets) > 1:
        lengths = [len(d) for d in datasets]
        # Each sample in dataset i gets weight 1/len_i so each dataset sums to 1
        weights = []
        for li in lengths:
            wi = 1.0 / max(li, 1)
            weights.extend([wi] * li)
        sampler = WeightedRandomSampler(weights, num_samples=sum(lengths), replacement=True)
        return DataLoader(concat, batch_size=kwargs['batch_size'], sampler=sampler, shuffle=False)

    return DataLoader(concat, batch_size=kwargs['batch_size'], shuffle=train)
   
def _get_preprocess_pipeline(train):
    if train:
        gamma_transform = transforms.Lambda(add_gamma)
        occlusion_transform = transforms.Lambda(add_occlusion)
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([gamma_transform], p=0.3),
            transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.0))], p=0.3),
            transforms.RandomApply([occlusion_transform], p=0.1)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    
def get_kitti(data_path, size, train_sequences, val_sequences, max_skip, train):
    from kitti import KITTI
    preprocess = _get_preprocess_pipeline(train)
    return KITTI(os.path.join(data_path, 'sequences'), 
                os.path.join(data_path, 'poses'), 
                transform=preprocess, 
                sequences=train_sequences if train else val_sequences,
                resize=size,
                apply_rcr=train,
                max_skip=max_skip if train else 0)
    

def get_queenscamp(data_path, size, train_sequences, val_sequences, max_skip, train):
    from queenscamp import QueensCAMP
    preprocess = _get_preprocess_pipeline(train)
    return QueensCAMP(data_path=data_path,
                      resize=size,
                      apply_rcr=train,
                      max_skip=max_skip if train else 0,
                      sequences=train_sequences if train else val_sequences, 
                      transform=preprocess)

def get_tartanair(data_path, size, train_sequences, val_sequences, max_skip, train):
    from tartanair import TartanAir

    preprocess = _get_preprocess_pipeline(train)
    train_environments = train_sequences
    val_environments = val_sequences

    train_sequences = []
    for env in train_environments:
        if os.path.exists(os.path.join(data_path, env, 'Easy')):
            folders = os.listdir(os.path.join(data_path, env, 'Easy'))
            train_sequences += [os.path.join(env, 'Easy', folder) for folder in folders]
        if os.path.exists(os.path.join(data_path, env, 'Hard')):
            folders = os.listdir(os.path.join(data_path, env, 'Hard'))
            train_sequences += [os.path.join(env, 'Hard', folder) for folder in folders]
                
    val_sequences = []
    for env in val_environments:
        if os.path.exists(os.path.join(data_path, env, 'Easy')):
            folders = os.listdir(os.path.join(data_path, env, 'Easy'))
            val_sequences += [os.path.join(env, 'Easy', folder) for folder in folders]
        if os.path.exists(os.path.join(data_path, env, 'Hard')):
            folders = os.listdir(os.path.join(data_path, env, 'Hard'))
            val_sequences += [os.path.join(env, 'Hard', folder) for folder in folders]

    return TartanAir(data_path=data_path,
                     sequences=train_sequences if train else val_sequences,
                     resize=size,
                     apply_rcr=train,
                     max_skip=max_skip if train else 0,
                     transform=preprocess)
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    loader = get_iterator(
        datasets_names=['tartanair','kitti', 'queenscamp'],
        train=True,
        kitti = {
            'data_path': '/home/hudson/Desktop/Unicamp/datasets/kitti',
            'size': (640, 640),
            'train_sequences': ["01","08"],
            'val_sequences': ["03"],
            'max_skip': 0
        },
        queenscamp = {
            'data_path': '/home/hudson/Desktop/Unicamp/datasets/queenscamp',
            'train_sequences': ["16"],
            'val_sequences': ["16"],
            'size': (640, 640),
            'max_skip': 0
        },
        tartanair = {
            'data_path': '/home/hudson/Desktop/Unicamp/datasets/tartanair',
            'train_sequences': ["abandonedfactory"],
            'val_sequences': [""],
            'size': (640, 640),
            'max_skip': 0
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
        