import torch
import numpy as np
from omegaconf import OmegaConf
from model import get_photo_vo_model

#from modvo.vo.tracker import Tracker

# class PhotoVOTracker(Tracker):
#     def __init__(self, **params):
#         self.photo_vo = None
#         self.first_image = True
#         self.index = 0
#         self.img0, self.img1 = None, None

#     def load_model(self, model_path):
#         self.photo_vo = torch.load(model_path)
#         self.photo_vo.eval()

#     def track(self, image):
#         if(self.index == 0):
#             self.R = np.identity(3)
#             self.t = np.zeros((3, 1))
#             self.img0 = image
#         else:
#             self.img1 = image
#             data = {}
#             data['view0']['image'] = self.img0
#             data['view1']['image'] = self.img1
#             vo = self.photo_vo(data)
#             print(vo)
#             #self.t = self.t + self.R.dot(t)
#             #self.R = R.dot(self.R)
#             self.img0 = self.img1
            
#         self.index += 1

#         return self.R, self.t
    
    
if __name__ == '__main__':
    model_path = 'best_model.tar'
    cp = torch.load(model_path)
    photo_vo_model = get_photo_vo_model(OmegaConf.create(cp["conf"]))
    photo_vo_model.load_state_dict(cp["model"], strict=False)
    photo_vo_model.eval()
    
    img0 = torch.rand(1, 3, 256, 256)
    img1 = torch.rand(1, 3, 256, 256)
    data = {}
    data['view0'] = {}
    data['view1'] = {}
    data['view0']['image'] = img0
    data['view1']['image'] = img1
    vo = photo_vo_model(data)['pred_vo']
    print(vo)

    # dlparams = {'root_path': '/root/datasets/kitti/',
    #             'start_frame': 0,
    #             'stop_frame': 800,
    #             'sequence_name': '03',
    #             'camera_id': '0',}
    # dataloader = KITTILoader(**dlparams)

    # det_params = {'nfeatures': 1000,
    #               'scaleFactor': 1.2,
    #               'nlevels': 8}
    # det = orb.ORBDetector(**det_params)

    # matcher_params = {'normType':cv2.NORM_L1, 
    #                   'crossCheck': False}
    # matcher = BFMatcher(**matcher_params)
    
    # vo_params = {'camera': dataloader.camera,
    #              'detector': det,
    #              'matcher': matcher}
    # vo = FrameByFrameTracker(**vo_params)
    # gui = GUIDrawer()
    # frames = []
    # for i, img  in enumerate(dataloader):
    #     print(i,'/', len(dataloader))
    #     R, t = vo.track(img)
    #     f = Frame(img)
    #     frame_pose = np.eye(4)
    #     frame_pose[:3,:3] = R
    #     frame_pose[:3,3] = t.flatten()
    #     f.pose = frame_pose
    #     frames.append(f)
    #     gui.draw_trajectory(frames)