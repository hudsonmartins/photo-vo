import os
import torch
import argparse
import cv2
import numpy as np
from skimage import io
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import MegaDepthDataset
from models.superpoint import SuperPoint
from models.superglue_patches import SuperGlue
from ground_truth import get_patches_points, get_ground_truth, get_pixels_from_pts
from models.utils import make_matching_plot_fast
from loss import nll_loss
import matplotlib.cm as cm

torch.set_grad_enabled(False)
PATCH_SIZE = 10
def pad_data(data, max_kpts, img_shape, device):
    for k in data:
        if isinstance(data[k], (list, tuple)):
            new_data = []
            if(k.startswith('keypoints')):
                #padding keypoints
                for kpt in data[k]:
                    new_data += [torch.cat((kpt, 
                                            torch.FloatTensor(max_kpts - kpt.shape[0], 2).
                                            uniform_(0, img_shape[1]).to(device)), 0)]
                    
            if(k.startswith('descriptor')):
                #padding descriptors
                for desc in data[k]:
                    new_data += [F.pad(desc, 
                                (0, max_kpts - desc.shape[1]))]

            if(k.startswith('score')):
                #padding scores
                for score in data[k]:
                    new_data += [F.pad(score, 
                                (0, max_kpts - score.shape[0]))]
            data[k] = torch.stack(new_data)
    return data

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

def create_kpts_image(img, kpts, color=(255,255,255)):
    for k in kpts:
        img = cv2.circle(img, (int(k[1]), int(k[0])), 3, color, 2)
    return img

def test(model_path, test_dataloader, device, config):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    superpoint = SuperPoint(config.get('superpoint', {})).to(device)   
    superglue = SuperGlue(config.get('superglue', {})).to(device)
    #superglue.load_state_dict(checkpoint['superglue'])
    nllloss = torch.nn.NLLLoss()

    test_pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    for batch_idx, batch in test_pbar:
        #Extract kpts
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
        data = pad_data(data, config['superpoint']['max_keypoints'], img0_gray.shape, device)

        #Forward
        ground_truth = get_ground_truth(data['keypoints0'], 
                                        data['keypoints1'], 
                                        batch, device, plot_vis=False)
        #loss = nllloss(scores, ground_truth)
        patches_pts0 = get_patches_points(data['keypoints0'])
        patches_pts1 = get_patches_points(data['keypoints1'])

        img0_pad = F.pad(img0, (PATCH_SIZE//2, PATCH_SIZE//2, PATCH_SIZE//2, PATCH_SIZE//2), "reflect")
        img1_pad = F.pad(img1, (PATCH_SIZE//2, PATCH_SIZE//2, PATCH_SIZE//2, PATCH_SIZE//2), "reflect")

        patches0 = get_pixels_from_pts(img0_pad, patches_pts0, output_shape=(patches_pts0.shape[0], patches_pts0.shape[1], PATCH_SIZE, PATCH_SIZE, 3))
        patches1 = get_pixels_from_pts(img1_pad, patches_pts1, output_shape=(patches_pts1.shape[0], patches_pts1.shape[1], PATCH_SIZE, PATCH_SIZE, 3))

        patches = {'patches0': patches0.to(device), 'patches1': patches1.to(device)}
        data = {**data, **patches}
        scores = superglue(data)

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > config['superglue']['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
                
        for i in range(ground_truth.size(0)):
            #loss = nllloss(scores[i], ground_truth[i])
            loss = nll_loss(scores, ground_truth, pos_weight=0.5, 
                                    neg_weight=0.5, log_prob=True)
            gt = ground_truth[i].cpu().numpy()
            ids1 = [x for x in range(len(gt)) if gt[x] < 400]
            ids2 = [x for x in gt if x < 400]
            
            kpts_img1 = create_kpts_image(img0.cpu().numpy()[i][0]*255, data['keypoints0'][i].cpu().numpy()[ids1])
            kpts_img2 = create_kpts_image(img1.cpu().numpy()[i][0]*255, data['keypoints1'][i].cpu().numpy()[ids2])
            gt_img = np.concatenate((kpts_img1, kpts_img2), axis=1)
            for (p1, p2) in zip(data['keypoints0'][i].cpu().numpy()[ids1], data['keypoints1'][i].cpu().numpy()[ids2]):
                gt_img = cv2.line(gt_img, (int(p1[1]), int(p1[0])), (int(p2[1]+kpts_img1.shape[0]), int(p2[0])), 255, 2)
            io.imsave('gdrive/My Drive/Doutorado/gt_image_'+str(batch_idx)+'_'+str(i)+'.png', gt_img)
            
            matches, conf = indices0.detach().cpu().numpy()[i], mscores0.detach().cpu().numpy()[i]
            valid = matches > -1
            mkpts0 = data['keypoints0'][i][valid]
            mkpts1 = data['keypoints1'][i][matches[valid]]
            mconf = conf[valid]
            color = cm.jet(mconf)

            text = ['SuperGlue',
                    'Keypoints: {}:{}'.format(len(data['keypoints0'][i]), len(data['keypoints1'][i])),
                    'Matches: {}'.format(len(mkpts0)),
                    'Loss: {}'.format(loss.item())]
            
            make_matching_plot_fast(img0.cpu().numpy()[i][0]*255, img1.cpu().numpy()[i][0]*255, 
                                data['keypoints0'][i].cpu().numpy(), data['keypoints1'][i].cpu().numpy(),
                                mkpts0.cpu().numpy(), mkpts1.cpu().numpy(), color, text, 
                                path='gdrive/My Drive/Doutorado/match_'+str(batch_idx)+'_'+str(i)+'.png', show_keypoints=True)
            
            
def main(dataset_path, test_scenes_path, model_path, batch_size, save_dataset_path, load_dataset_from_file):
    
    config = {'superpoint': {'nms_radius': 4,
                            'keypoint_threshold': 0.005,
                            'max_keypoints': 400},
              'superglue': {'weights': 'outdoor',
                            'sinkhorn_iterations': 20,
                            'match_threshold': 0.5}}

    scenes_info_path = os.path.join(dataset_path, 'scene_info')
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")                              

    random_seed = 735
    #Seed
    torch.manual_seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    test_dset = MegaDepthDataset(scene_list_path=test_scenes_path,
                                scene_info_path=scenes_info_path,
                                base_path=dataset_path,
                                train=True,
                                preprocessing='torch',
                                min_overlap_ratio=0.1,
                                max_overlap_ratio=0.7,
                                image_size=720,
                                save_dataset_path=os.path.join(save_dataset_path, "train_dset.pkl"),
                                load_from_file=load_dataset_from_file)

    test_dataloader = DataLoader(test_dset, batch_size=batch_size, num_workers=2)
    test_dset.build_dataset()
    test(model_path, test_dataloader, device, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", default=None, help="Path to dataset")
    parser.add_argument("test_scenes_path", default=None, help="Path to train scenes txt")
    parser.add_argument("--model_path", default="models/model_0.pth", help="Path to save checkpoints")   
    parser.add_argument("--save_dataset_path", default="logs/", help="Path to save built dataset")   
    parser.add_argument("--batch_size", default=1, type=int, help="Batch Size")
    parser.add_argument("--load_dataset_from_file", action='store_true', help="True if we should load the dataset from a pickle file")   
    args = parser.parse_args()

    main(args.dataset_path, args.test_scenes_path, args.model_path, args.batch_size, args.save_dataset_path, args.load_dataset_from_file)