import os
import sys
import yaml
import torch
import importlib
import argparse
import numpy as np
from omegaconf import OmegaConf
from model import get_photo_vo_model
from utils import euler_angles_to_matrix
from gluefactory.utils.image import numpy_image_to_torch, ImagePreprocessor

from modvo.vo.tracker import Tracker

class PhotoVOTracker(Tracker):
    def __init__(self, **params):
        self.first_image = True
        self.index = 0
        self.img0, self.img1 = None, None
        self.photo_vo_model = self.load_model(params['path'])

    def load_model(self, model_path):
        cp = torch.load(model_path)
        model = get_photo_vo_model(OmegaConf.create(cp["conf"]))
        model.load_state_dict(cp["model"], strict=False)
        model.eval()
        self.preprocessor = ImagePreprocessor(OmegaConf.create(cp["conf"]["data"]["preprocessing"]))
        return model
    
    def get_input(self):
        im0_torch = numpy_image_to_torch(self.img0)
        im1_torch = numpy_image_to_torch(self.img1)
        im0 = self.preprocessor(im0_torch)['image']
        im1 = self.preprocessor(im1_torch)['image']
        return {
                'view0': {
                    'image': torch.unsqueeze(im0, 0),
                },
                'view1': {
                    'image': torch.unsqueeze(im1, 0),
                }
            }

    
    def track(self, image):
        if(self.index == 0):
            self.R = np.identity(3)
            self.t = np.zeros((3, 1))
            self.img0 = image
        else:
            self.img1 = image
            data = self.get_input()
            vo = self.photo_vo_model(data)['pred_vo'][0]
            t = vo[:3].reshape(3, 1).detach().cpu().numpy()
            R = euler_angles_to_matrix(vo[3:], "XYZ").reshape(3, 3).detach().cpu().numpy()
            
            self.t = self.t + self.R.dot(t)
            self.R = R.dot(self.R)
            self.img0 = self.img1
            
        self.index += 1

        return self.R, self.t

def main(args):
    with open(args.pipeline_config, 'r') as f:
        config = yaml.safe_load(f)

    #loading classes
    dloader_class = config['dataloader']['class']
    print('Dataloader %s' % dloader_class)
    module = importlib.import_module('modvo.dataloaders.'+dloader_class.rsplit('.',1)[0])
    attr = getattr(module, dloader_class.rsplit('.', 1)[-1])
    #get params without class name
    params = {k: v for k, v in config['dataloader'].items() if k != 'class'}
    dataloader = attr(**params)
    
    params = {k: v for k, v in config['model'].items() if k != 'class'}
    vo = PhotoVOTracker(**params)

    os.makedirs(args.output_path, exist_ok=True)
    log_fopen = open(os.path.join(args.output_path, args.trajectory_file), mode='a')
    print('Enable GUI: ', args.enable_gui)

    if args.enable_gui:
        import numpy as np
        from modvo.maps.kf_based import Frame
        from modvo.gui.viewer import GUIDrawer
        drawer = GUIDrawer()
        frames = []
    
    if args.output_format == 'tum':
        from modvo.utils.geometry import matrix_to_quaternion
   
    while dataloader.is_running:
        print("-"*50)
        try:
            image = next(dataloader)    
        except StopIteration:
            print("Finishing...")
            break
        if(image is None):
            continue
        print('img shape ', image.shape)
        R, t = vo.track(image)
        
        if args.enable_gui:
            f = Frame(image)
            frame_pose = np.eye(4)
            frame_pose[:3,:3] = R
            frame_pose[:3,3] = t.flatten()
            f.pose = frame_pose
            frames.append(f)
            drawer.draw_trajectory(frames)
        if(dataloader.type == 'dataset'):
            i = dataloader.index
            print(i,'/', len(dataloader))
        else:
            print('frame ', dataloader.index)
        
        if args.output_format == 'kitti':
            print(R[0, 0], R[0, 1], R[0, 2], t[0, 0],
                 R[1, 0], R[1, 1], R[1, 2], t[1, 0],
                R[2, 0], R[2, 1], R[2, 2], t[2, 0],
                file=log_fopen)
        elif args.output_format == 'tum':
            timestamp = dataloader.get_timestamp()
            q = matrix_to_quaternion([[R[0, 0], R[0, 1], R[0, 2]],
                                        [R[1, 0], R[1, 1], R[1, 2]],
                                        [R[2, 0], R[2, 1], R[2, 2]]])
            print(str(timestamp), t[0, 0], t[1, 0], t[2, 0], q[0], q[1], q[2], q[3],
                file=log_fopen)

    sys.exit(0)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pipeline_config', type=str, help='Path to the pipeline configuration file')
    parser.add_argument('--output_path', type=str, default = '/root/modvo/results/', help='path to save all outputs')
    parser.add_argument('--trajectory_file', type=str, default = 'trajectory.txt', help='name of the trajectory file')
    parser.add_argument('--output_format', type=str, default = 'kitti', help='file format to save trajectory (either kitti or tum)')
    parser.add_argument('--enable_gui', action='store_true', help='use this flag to enable gui')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)