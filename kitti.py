import torch
import numpy as np
from utils import kitti_to_6dof
from PIL import ImageFile, Image
from pathlib import Path
from torchvision import transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

#kitti pose mean and std
KITT_MEAN = [-8.6736e-5, -1.6038e-2, 9.0033e-1, 1.7061e-5, 9.5582e-4, -5.5258e-5]
KITTI_STD = [2.5584e-2, 1.8545e-2, 3.0352e-1, 2.8256e-3, 1.7771e-2, 3.2326e-3]

def get_iterator(data_path, size, batch_size, sequences_names, max_skip):
    random_seed = 42
    rand = np.random.RandomState(random_seed)
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return Dataset(preprocess, data_path, batch_size=batch_size,
                   rand=rand, sequences_names=sequences_names, max_skip=max_skip)


class Dataset():
    def __init__(self, preprocess, data_path, batch_size, rand, sequences_names,
                 max_skip=5):
        self.preprocess = preprocess
        self._ys = [self._load_poses(data_path, sequence) for sequence in sequences_names]
        self._batch_size = batch_size
        self._rand = rand
        self._rand_state = self._rand.get_state()
        self._data_path = Path(data_path)
        self.sequences_names = sequences_names
        self._n_iterations = 0
        self._sequences = list(range(len(self._ys)))
        self._max_skip = max_skip

    def _load_poses(self, data_path, sequence):
        pose_file = Path(data_path) / 'poses' / f'{sequence}.txt'
        poses = np.loadtxt(pose_file, dtype=np.float32).reshape(-1, 12)
        return poses

    def reset(self):
        self._rand.set_state(self._rand_state)
        self._n_iterations = 0

    def read_image(self, path):
        return Image.open(path).convert('RGB')

    def _load_images(self, sequence, index, pair_index):
        base_path = self._data_path / 'sequences' / self.sequences_names[sequence] / 'image_0'
        img1 = self.read_image(base_path / f'{index:06d}.png')
        img1 = self.preprocess(img1)
        img2 = self.read_image(base_path / f'{pair_index:06d}.png')
        img2 = self.preprocess(img2)
        return img1, img2

    def get_batch(self):
        imgs, ys = [], []
        for _ in range(self._batch_size):
            sequence = int(self._rand.choice(self._sequences))
            index = self._rand.choice(len(self._ys[sequence]))
            pair_index = min(index + self._rand.choice(range(1, self._max_skip + 1)), len(self._ys[sequence]) - 1)
            # Loading images
            x1, x2 = self._load_images(sequence, index, pair_index)
            
            # Loading and normalizing poses
            pose1 = np.vstack([self._ys[sequence][index].reshape(3, 4), [[0., 0., 0., 1.]]])
            pose2 = np.vstack([self._ys[sequence][pair_index].reshape(3, 4), [[0., 0., 0., 1.]]])

            pose2_to_1 = np.dot(np.linalg.inv(pose1), pose2)
            y = kitti_to_6dof(pose2_to_1[:3].reshape(-1))
            
            #normalize
            y = (y - KITT_MEAN) / KITTI_STD
            
            imgs.append(torch.stack([x1, x2]))
            ys.append(y)

        self._n_iterations += 1
        imgs = np.stack(imgs)
        ys = np.stack(ys)
        return torch.from_numpy(imgs).float(), torch.from_numpy(ys).float()

    def iterate(self):
        while True:
            yield self.get_batch()


if __name__ == '__main__':
    from skimage import io
    import matplotlib
    matplotlib.use('TkAgg')  # Switch to an interactive backend like TkAgg

    it = get_iterator('/home/hudson/Desktop/Unicamp/datasets/kitti/', size=(512,512),
                      batch_size=1, sequences_names=['03'], max_skip=1)
    for x, y in it.iterate():
        print(x.shape, y)
        print(x)
        #io.imshow(x[0, 0].transpose(1, 2, 0))
        #io.show()
        #io.imshow(x[0, 1].transpose(1, 2, 0))
        #io.show()
        break