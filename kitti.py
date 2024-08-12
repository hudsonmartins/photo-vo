import torch
import numpy as np
from utils import kitti_to_6dof
from PIL import ImageFile, Image
from pathlib import Path
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_iterator(data_path, size, cycle_every, batch_size, sequences_names, max_skip):
    random_seed = 42
    rand = np.random.RandomState(random_seed)
    return Dataset(data_path, batch_size=batch_size, image_size=(size, size),
                   rand=rand, sequences_names=sequences_names, max_skip=max_skip,
                   cycle_every=cycle_every)


class Dataset():
    def __init__(self, data_path, batch_size, image_size, rand, sequences_names,
                 max_skip=5, cycle_every=None):
        
        self._ys = [self._load_poses(data_path, sequence) for sequence in sequences_names]
        self._batch_size = batch_size
        self._rand = rand
        self._rand_state = self._rand.get_state()
        self._data_path = Path(data_path)
        self.sequences_names = sequences_names
        self._cycle_every = cycle_every
        self._n_iterations = 0
        self._sequences = list(range(len(self._ys)))
        self._img_size = image_size
        self._max_skip = max_skip

    def _load_poses(self, data_path, sequence):
        pose_file = Path(data_path) / 'poses' / f'{sequence}.txt'
        poses = np.loadtxt(pose_file, dtype=np.float32).reshape(-1, 12)
        return poses

    def reset(self):
        self._rand.set_state(self._rand_state)
        self._n_iterations = 0

    def read_image(self, path):
        img = Image.open(path).convert('RGB')
        img = img.resize(self._img_size, Image.LANCZOS)
        return np.array(img)

    def _load_images(self, sequence, index, pair_index):
        base_path = self._data_path / 'sequences' / self.sequences_names[sequence] / 'image_0'
        img1 = self.read_image(base_path / f'{index:06d}.png')
        img2 = self.read_image(base_path / f'{pair_index:06d}.png')
        return img1, img2

    def get_batch(self):
        if self._cycle_every and self._n_iterations and self._n_iterations % self._cycle_every == 0:
            self.reset()

        imgs, ys = [], []
        for _ in range(self._batch_size):
            sequence = int(self._rand.choice(self._sequences))
            index = self._rand.choice(len(self._ys[sequence]))
            pair_index = min(index + self._rand.choice(range(1, self._max_skip + 1)), len(self._ys[sequence]) - 1)
            # Loading images
            x1, x2 = self._load_images(sequence, index, pair_index)

            # Convert to channels first and normalize
            x1 = np.rollaxis(x1, -1, 0) / 255.0
            x2 = np.rollaxis(x2, -1, 0) / 255.0
            
            # Loading and normalizing poses
            pose1 = np.vstack([self._ys[sequence][index].reshape(3, 4), [[0., 0., 0., 1.]]])
            pose2 = np.vstack([self._ys[sequence][pair_index].reshape(3, 4), [[0., 0., 0., 1.]]])

            pose2_to_1 = np.dot(np.linalg.inv(pose1), pose2)
            y = kitti_to_6dof(pose2_to_1[:3].reshape(-1))
            
            imgs.append(np.array([x1, x2]))
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

    it = get_iterator('/home/hudson/Desktop/Unicamp/datasets/kitti/', size=512, cycle_every=None, batch_size=1,
                      sequences_names=['03'], max_skip=5)
    for x, y in it.iterate():
        print(x.shape, y)
        io.imshow(x[0, 0].transpose(1, 2, 0))
        io.show()
        io.imshow(x[0, 1].transpose(1, 2, 0))
        io.show()
        break