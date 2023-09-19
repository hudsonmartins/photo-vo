import os
import sys
import torch
import argparse
import multiprocessing
import matplotlib
import numpy as np
from tqdm import tqdm
from skimage import io
import matplotlib.cm as cm
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from dataset import MegaDepthDataset
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.motion_estimator import MotionEstimator

#from models.monodepth2 import MonoDepth
from loss import nll_loss, patches_photometric_loss, pose_error
from models.utils import make_matching_plot_fast
from ground_truth import get_ground_truth, get_patches_points, warp_patches, get_pixels_from_pts
from utils import *

matplotlib.use('TkAgg')

IMAGE_SIZE = 720
MAX_KPTS = 50
PATCH_SIZE = 10

def train(lr, num_epochs, save_every, pos_weight, neg_weight, train_dataloader, validation_dataloader, 
         load_model_path, max_iter, checkpoints_path, config, device, writer, w_matches=0.4, w_photo=0.3, w_vo=0.3,
         only_val=False, debug_mode=False):

    iter = 0
    superpoint = SuperPoint(config.get('superpoint', {})).to(device)
    superglue = SuperGlue(config.get('superglue', {})).to(device)
    motion_estimator = MotionEstimator(config.get('motion_estimator', {})).to(device)
    optimizer = optim.Adam(superglue.parameters(), lr=lr)
    start_epoch = 1
    start_step = 0
    match_loss = None
    best_loss = None
    if(load_model_path == None):
        path = Path(__file__).parent
        path = path / 'models/weights/superglue_outdoor.pth'
        #superglue = load_model_weights(superglue, path)    
        superglue.load_state_dict(torch.load(path))
    else:
        superglue, start_epoch, start_step, optimizer_state, match_loss = load_model_weights(superglue, load_model_path, 
                                                                               recover_state=True,
                                                                               modules=['kenc', 'penc', 'gnn', 'final_proj'])
        optimizer.load_state_dict(optimizer_state)
        print('starting from epoch ', start_epoch)
        print('starting from step ', start_step)

        
    
    for param in superpoint.parameters():
        param.requires_grad = False
    for param in superglue.parameters():
        param.requires_grad = False

    for epoch_idx in range(start_epoch, num_epochs + 1):
        train_size = min(max_iter, len(train_dataloader))
        train_pbar = tqdm(enumerate(train_dataloader), total=train_size)
        if(validation_dataloader != None):
            val_size = min(max_iter, len(validation_dataloader))
            val_pbar = tqdm(enumerate(validation_dataloader), total=val_size)
        training_losses = []
        matching_losses = []
        photometric_losses = []
        vo_losses = []
        validation_losses = []
        matches_matrix = None      
        if(not only_val):
            print('\n')
            print('='*20)
            print('Training...')
            if(epoch_idx == start_epoch and start_step > max_iter):
                continue
            
            for batch_idx, batch in train_pbar:
                if(batch_idx >= max_iter):
                    break
                if(batch_idx < start_step and epoch_idx == start_epoch):
                    continue
                
                optimizer.zero_grad()
                img0 = batch['image1'].to(device)
                img1 = batch['image2'].to(device)
                img0_gray = batch['gray1'].to(device)
                img1_gray = batch['gray2'].to(device)
            
                kpts = {}
                sp1 = superpoint({'image': img0_gray})
                kpts = {**kpts, **{k+'0': v for k, v in sp1.items()}}
                sp2 = superpoint({'image': img1_gray})
                kpts = {**kpts, **{k+'1': v for k, v in sp2.items()}}

                data = {'image0': img0_gray, 'image1': img1_gray}
                data = {**data, **kpts}
                
                data = pad_data(data, config['superpoint']['max_keypoints'], 
                                img0_gray.shape, device)

                gt_matches = get_ground_truth(data['keypoints0'], 
                                             data['keypoints1'], 
                                             batch, device)

                patches_pts0 = get_patches_points(data['keypoints0'])
                patches_pts1 = get_patches_points(data['keypoints1'])                                     
                
                img0_pad = F.pad(img0, (PATCH_SIZE//2, PATCH_SIZE//2, PATCH_SIZE//2, PATCH_SIZE//2), "reflect")
                img1_pad = F.pad(img1, (PATCH_SIZE//2, PATCH_SIZE//2, PATCH_SIZE//2, PATCH_SIZE//2), "reflect")

                patches0 = get_pixels_from_pts(img0_pad, patches_pts0, output_shape=(patches_pts0.shape[0], patches_pts0.shape[1], PATCH_SIZE, PATCH_SIZE, 3))
                patches1 = get_pixels_from_pts(img1_pad, patches_pts1, output_shape=(patches_pts1.shape[0], patches_pts1.shape[1], PATCH_SIZE, PATCH_SIZE, 3))

                patches = {'patches0': patches0.to(device), 'patches1': patches1.to(device)}
                data = {**data, **patches}
                             
                if(gt_matches == None):
                    continue

                matches = superglue(data) #forward to superglue
                match_loss = nll_loss(matches['scores_matrix'], gt_matches, pos_weight=pos_weight, 
                                      neg_weight=neg_weight)
                
                data = {**data, 'image0': img0, 'image1': img1}
                data = {**data, **{'matches': matches['scores_matrix'].to(device)}}
                vo = motion_estimator(data) #forward to motion estimation
                vo_loss = pose_error(vo, batch['pose1'].to(device), batch['pose2'].to(device))

                vo_matrix = torch.zeros((batch['image1'].shape[0], 4,4)) #batch_size, 4, 4
                vo_matrix[...,:3,:3] = euler_angles_to_matrix(vo[..., 3:], convention="ZYX")
                vo_matrix[...,:3,3] = vo[...,  :3]
                vo_matrix[...,3,3] = 1
                vo_matrix = torch.Tensor(vo_matrix).to(device)

                patches_pts0, pts0_1 = warp_patches(patches_pts0.to(device), batch['depth1'].to(device), batch['intrinsics1'].to(device), 
                                                    batch['bbox1'].to(device), batch['depth2'].to(device), batch['intrinsics2'].to(device), 
                                                    batch['bbox2'].to(device), vo_matrix, PATCH_SIZE, MAX_KPTS)               
                                
                patches0_1 = get_pixels_from_pts(img1, pts0_1, output_shape=(pts0_1.shape[0], pts0_1.shape[1], PATCH_SIZE, PATCH_SIZE, 3)) 
                matches_matrix, scores = scores_to_matches(matches['scores_matrix'], config['superglue']['match_threshold'])
                photometric_loss = patches_photometric_loss(patches0_1.to(device), patches1.to(device), matches_matrix.to(device))

                #draw patches
                if(debug_mode):
                    
                    viz0 = np.transpose(data['image0'][0].cpu().numpy(), (1,2,0))
                    viz1 = np.transpose(data['image1'][0].cpu().numpy(), (1,2,0))
                    viz0 = np.ascontiguousarray(viz0)
                    viz1 = np.ascontiguousarray(viz1)
                    
                    for i in range(patches_pts0.shape[1]):
                        viz0 = draw_pts(viz0, patches_pts0[0,i].cpu().numpy(), color=(0,255,0))
                        viz1 = draw_pts(viz1, pts0_1[0,i].cpu().numpy(),color=(255,0,0))
                        viz1 = draw_pts(viz1, patches_pts1[0,i].cpu().numpy(), color=(0,255,0))
                        
                        matches_img = draw_matches(viz0, viz1, data['keypoints0'][0].cpu().numpy(), 
                                                data['keypoints1'][0].cpu().numpy(), 
                                                matches_matrix[0].cpu().numpy(), 
                                                scores[0].cpu().numpy(), channels_first=False, draw_kpts=True)

                        cv2.putText(matches_img, 'photometric loss: '+str(photometric_loss.item()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.putText(matches_img, 'match loss: '+str(0), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.putText(matches_img, 'vo loss: '+str(vo_loss.item()), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        #io.imshow(matches_img)
                        #io.show()
                
                if(match_loss != None and photometric_loss != None and vo_loss != None):
                    loss = w_matches * match_loss + w_photo * photometric_loss + w_vo * vo_loss
                    loss.backward()
                    optimizer.step()
                    current_loss = loss.item()
                    training_losses.append(current_loss)
                    matching_losses.append(match_loss.item())
                    photometric_losses.append(photometric_loss.item())
                    vo_losses.append(vo_loss.item())
                    train_pbar.set_postfix(loss=('%.4f' % np.mean(training_losses)))
                    
                    if(batch_idx%save_every == 0):
                        output_name = f'model_{epoch_idx}_{batch_idx}'
                        save_model(os.path.join(checkpoints_path, output_name+".pth"), 
                                superglue, optimizer, batch_idx, epoch_idx, loss)
                    iter+=1
                    
            if(matches_matrix != None):
                #Adding predictions to tensorboard
                m_tb = matches_matrix.detach().cpu().numpy()
                sc_tb = scores.detach().cpu().numpy()
                imgs0_cpu = [im0.cpu().numpy() for im0 in data['image0']]
                imgs1_cpu = [im1.cpu().numpy() for im1 in data['image1']]
                kpts0_cpu = [k0.cpu().numpy() for k0 in data['keypoints0']]
                kpts1_cpu = [k1.cpu().numpy() for k1 in data['keypoints1']]

                m_imgs = []
                for im0, im1, k0, k1, m, s in zip(imgs0_cpu, imgs1_cpu, kpts0_cpu, kpts1_cpu, m_tb, sc_tb):
                    matches_img = draw_matches(np.transpose(im0, (1,2,0)), np.transpose(im1, (1,2,0)),
                                                k0, k1, m, s, channels_first=False, draw_kpts=True)
                    m_imgs.append(matches_img)
                    
                m_imgs = torch.from_numpy(np.array(m_imgs)).permute(0, 3, 1, 2)
                imgs_grid = vutils.make_grid(m_imgs)
                writer.add_image('train/pred', imgs_grid, epoch_idx)

                #Adding ground truth matches to tensorboard
                gt_m_tb, gt_sc_tb = scores_to_matches(gt_matches, config['superglue']['match_threshold'])
                gt_m_tb = gt_m_tb.detach().cpu().numpy()
                gt_sc_tb = gt_sc_tb.detach().cpu().numpy()
                gt_m_imgs = []
                for im0, im1, k0, k1, m, s in zip(imgs0_cpu, imgs1_cpu, kpts0_cpu, kpts1_cpu, gt_m_tb, gt_sc_tb):
                    matches_img = draw_matches(np.transpose(im0, (1,2,0)), np.transpose(im1, (1,2,0)),
                                                k0, k1, m, s, channels_first=False, draw_kpts=True)
                    gt_m_imgs.append(matches_img)

                gt_m_imgs = torch.from_numpy(np.array(gt_m_imgs)).permute(0, 3, 1, 2)
                gt_imgs_grid = vutils.make_grid(gt_m_imgs)
                writer.add_image('train/gt_matches', gt_imgs_grid, epoch_idx)
                writer.add_scalar('Training_loss ', np.mean(training_losses), epoch_idx)
                writer.add_scalar('Matching_loss ', np.mean(matching_losses), epoch_idx)
                writer.add_scalar('Photometric_loss ', np.mean(photometric_losses), epoch_idx)
                writer.add_scalar('VO_loss ', np.mean(vo_losses), epoch_idx)
                writer.flush()

        if(validation_dataloader != None):
            print('\n')
            print('='*20)
            print('Validation...')
            #Validation Loop
            superpoint.eval()
            superglue.eval()
            motion_estimator.eval()

            with torch.no_grad():
                gt_m_imgs = []
                m_imgs = []
                for batch_idx, batch in val_pbar:
                    if(batch_idx >= max_iter):
                        break
                    img0 = batch['image1'].to(device)
                    img1 = batch['image2'].to(device)
                    img0_gray = batch['gray1'].to(device)
                    img1_gray = batch['gray2'].to(device)
                    kpts = {}
                    sp1 = superpoint({'image': img0_gray})
                    kpts = {**kpts, **{k+'0': v for k, v in sp1.items()}}
                    sp2 = superpoint({'image': img1_gray})
                    kpts = {**kpts, **{k+'1': v for k, v in sp2.items()}}
                    data = {'image0': img0_gray, 'image1': img1_gray}
                    data = {**data, **kpts}
                    data = pad_data(data, config['superpoint']['max_keypoints'],
                                    img0_gray.shape, device)
                    gt_matches = get_ground_truth(data['keypoints0'],
                                                data['keypoints1'],
                                                batch, device)
                    #matches
                    matches = superglue(data) #forward to superglue
                    match_loss = nll_loss(matches['scores_matrix'], gt_matches, pos_weight=pos_weight, 
                                      neg_weight=neg_weight)
                
                    data = {**data, 'image0': img0, 'image1': img1}
                    data = {**data, **{'matches': matches['scores_matrix'].to(device)}}
                    vo = motion_estimator(data) #forward to motion estimation
                    vo_loss = pose_error(vo, batch['pose1'].to(device), batch['pose2'].to(device))

                    vo_matrix = torch.zeros((batch['image1'].shape[0], 4,4)) #batch_size, 4, 4
                    vo_matrix[...,:3,:3] = euler_angles_to_matrix(vo[..., 3:], convention="ZYX")
                    vo_matrix[...,:3,3] = vo[...,  :3]
                    vo_matrix[...,3,3] = 1
                    vo_matrix = torch.Tensor(vo_matrix).to(device)

                    patches_pts0, pts0_1 = warp_patches(patches_pts0.to(device), batch['depth1'].to(device), batch['intrinsics1'].to(device), 
                                                        batch['bbox1'].to(device), batch['depth2'].to(device), batch['intrinsics2'].to(device), 
                                                        batch['bbox2'].to(device), vo_matrix, PATCH_SIZE, MAX_KPTS)               
                                    
                    patches0_1 = get_pixels_from_pts(img1, pts0_1, output_shape=(pts0_1.shape[0], pts0_1.shape[1], PATCH_SIZE, PATCH_SIZE, 3)) 
                    matches_matrix, scores = scores_to_matches(matches['scores_matrix'], config['superglue']['match_threshold'])
                    photometric_loss = patches_photometric_loss(patches0_1.to(device), patches1.to(device), matches_matrix.to(device))
                    loss = w_matches * match_loss + w_photo * photometric_loss + w_vo * vo_loss
                    current_loss = loss.item()
                    validation_losses.append(current_loss)
                    val_pbar.set_postfix(loss=('%.4f' % np.mean(validation_losses)))

                    if(batch_idx == 0 and matches != None):
                        #Adding predicted matches to tensorboard
                        m_tb = matches_matrix.detach().cpu().numpy()
                        sc_tb = scores.detach().cpu().numpy()
                        imgs0_cpu = [im0.cpu().numpy() for im0 in data['image0']]
                        imgs1_cpu = [im1.cpu().numpy() for im1 in data['image1']]
                        kpts0_cpu = [k0.cpu().numpy() for k0 in data['keypoints0']]
                        kpts1_cpu = [k1.cpu().numpy() for k1 in data['keypoints1']]

                        m_imgs = []
                        for im0, im1, k0, k1, m, s in zip(imgs0_cpu, imgs1_cpu, kpts0_cpu, kpts1_cpu, m_tb, sc_tb):
                            matches_img = draw_matches(np.transpose(im0, (1,2,0)), np.transpose(im1, (1,2,0)),
                                                        k0, k1, m, s, channels_first=False, draw_kpts=True)
                            m_imgs.append(matches_img)
                            
                        #Adding ground truth matches to tensorboard
                        gt_m_tb, gt_sc_tb = scores_to_matches(gt_matches, config['superglue']['match_threshold'])
                        gt_m_tb = gt_m_tb.detach().cpu().numpy()
                        gt_sc_tb = gt_sc_tb.detach().cpu().numpy()
                        gt_m_imgs = []
                        for im0, im1, k0, k1, m, s in zip(imgs0_cpu, imgs1_cpu, kpts0_cpu, kpts1_cpu, gt_m_tb, gt_sc_tb):
                            matches_img = draw_matches(np.transpose(im0, (1,2,0)), np.transpose(im1, (1,2,0)),
                                                        k0, k1, m, s, channels_first=False, draw_kpts=True)
                            gt_m_imgs.append(matches_img)
                        
            m_imgs = torch.from_numpy(np.array(m_imgs)).permute(0, 3, 1, 2)
            gt_m_imgs = torch.from_numpy(np.array(gt_m_imgs)).permute(0, 3, 1, 2)       
            
            imgs_grid = vutils.make_grid(m_imgs)
            writer.add_image('val/pred', imgs_grid, epoch_idx)
            gt_imgs_grid = vutils.make_grid(gt_m_imgs)
            writer.add_image('val/gt', gt_imgs_grid, epoch_idx)
            title = 'Validation_loss '
            writer.add_scalar(title, np.mean(validation_losses), epoch_idx)
            writer.flush()

        if(not only_val):
            if(best_loss == None or np.mean(validation_losses) < best_loss):
                best_loss = np.mean(validation_losses)
                torch.save(superpoint.state_dict(), os.path.join(checkpoints_path, f'superpoint_{epoch_idx}.pth'))
                output_name = f'superglue_{epoch_idx}'
                save_model(os.path.join(checkpoints_path, output_name+".pth"), 
                            superglue, optimizer, len(train_dataloader), epoch_idx, loss)
                output_name = f'motion_estimator_{epoch_idx}'
                torch.save(motion_estimator.state_dict(), os.path.join(checkpoints_path, output_name+".pth"))


def main(lr, batch_size, num_epochs, save_every, dataset_path, train_scenes_path, 
        load_model_path, valid_scenes_path, logs_dir, max_iter,
        checkpoints_path, save_dataset_path, load_dataset_from_file,
        pos_weight, neg_weight, only_val):
        
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)
    config = {'superpoint': {'nms_radius': 4,
                            'keypoint_threshold': 0.005,
                            'max_keypoints': MAX_KPTS},
              'superglue': {'weights': 'outdoor',
                            'sinkhorn_iterations': 30,
                            'match_threshold': 0.2},
              'motion_estimator': {'n_kpts': MAX_KPTS,}
            }

    scenes_info_path = os.path.join(dataset_path, 'scene_info')
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")                              

    random_seed = 33
    #Seed
    torch.manual_seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    train_dset = MegaDepthDataset(scene_list_path=train_scenes_path,
                            scene_info_path=scenes_info_path,
                            base_path=dataset_path,
                            preprocessing='torch',
                            min_overlap_ratio=0.1,
                            max_overlap_ratio=0.7,
                            image_size=IMAGE_SIZE,
                            save_dataset_path=os.path.join(save_dataset_path, "train_dset.pkl"),
                            load_from_file=load_dataset_from_file)

    val_dset = MegaDepthDataset(scene_list_path=valid_scenes_path,
                                scene_info_path=scenes_info_path,
                                base_path=dataset_path,
                                train=False,
                                preprocessing='torch',
                                min_overlap_ratio=0.1,
                                max_overlap_ratio=0.7,
                                image_size=IMAGE_SIZE,
                                save_dataset_path=os.path.join(save_dataset_path, "valid_dset.pkl"),
                                load_from_file=load_dataset_from_file)

    train_dataloader = DataLoader(train_dset, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
    train_dset.build_dataset()
    
    validation_dataloader = DataLoader(val_dset, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
    val_dset.build_dataset()

    writer = SummaryWriter(logs_dir,
        comment= "_LR_"+ str(lr) + "_Batch_size_" + str(batch_size))
    
    train(lr, num_epochs, save_every, pos_weight, neg_weight, 
          train_dataloader, validation_dataloader, load_model_path,
          max_iter, checkpoints_path, config, device, writer, 
          only_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", default=None, help="Path to dataset")
    parser.add_argument("train_scenes_path", default=None, help="Path to train scenes txt")
    parser.add_argument("valid_scenes_path", default=None, help="Path to valid scenes txt")
    parser.add_argument("--load_model_path", default=None, help="Path to load model")
    parser.add_argument("--logs_dir", default="logs/", help="Path to save logs")
    parser.add_argument("--max_iter", default=1000, type=int, help="Max training iterations")
    parser.add_argument("--checkpoints_path", default="models/", help="Path to save checkpoints")   
    parser.add_argument("--save_dataset_path", default="logs/", help="Path to save built dataset")   
    parser.add_argument("--load_dataset_from_file", action='store_true', help="True if we should load the dataset from a pickle file")   
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning Rate")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch Size")
    parser.add_argument("--num_epochs", default=100, type=int, help="Path to logs")
    parser.add_argument("--save_every", default=200, type=int, help="Save model after this number of iterations")
    parser.add_argument("--pos_weight", default=0.5, type=float, help="Weight to compute loss in positive samples")
    parser.add_argument("--neg_weight", default=0.5, type=float, help="Weight to compute loss in negative samples")
    parser.add_argument("--only_val", action='store_true')

    args = parser.parse_args()
    
    main(args.learning_rate, args.batch_size, args.num_epochs, args.save_every,
         args.dataset_path, args.train_scenes_path, args.load_model_path, 
         args.valid_scenes_path, args.logs_dir, args.max_iter, args.checkpoints_path, 
         args.save_dataset_path, args.load_dataset_from_file,
         args.pos_weight, args.neg_weight, args.only_val)