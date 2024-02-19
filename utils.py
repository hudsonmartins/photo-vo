import torch


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

