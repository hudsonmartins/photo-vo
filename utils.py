import torch

def draw_kpts(img, kpts, color=(0, 0, 255), radius=5):
    """Draw keypoints on an image"""
    for pt in kpts:
        img = cv2.circle(img, (pt[0], pt[1]), radius, color, 2)
    return img

def get_patches(img, kpts, patch_size=10):
    """Given an image and a set of keypoints, return the patches around the keypoints"""

    batch_size = img.size(0)
    patch_size = int(patch_size)
    half = patch_size // 2
    patches = []
    for i in range(batch_size):
        #pad image 
        img_pad = torch.nn.functional.pad(img[i], (half, half, half, half), mode='reflect')
        print('image after padding ', img_pad.size())
        kpts_i = kpts[i]
        patches_i = []
        for j in range(kpts_i.size(0)):
            x, y = kpts_i[j].int()
            print('x, y ', x, y)
            
            patch = img_pad[..., y:y+patch_size, x:x+patch_size]
            patches_i.append(patch)
        patches.append(torch.stack(patches_i))

    return torch.stack(patches)


if __name__ == "__main__":
    import cv2
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
