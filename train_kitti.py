import os
import torch
import glob
import argparse
import numpy as np
import random
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from gluefactory.geometry.wrappers import Pose

from loss import pose_loss_norm
from iterators import get_iterator
from model import get_photo_vo_model
from utils import batch_to_device, draw_camera_poses, draw_matches


# Setup logging
logger = logging.getLogger("kitti_train")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

# Default training configuration
default_train_conf = {
    "epochs": 10,
    "optimizer": "adam",
    "optimizer_options": {},
    "lr": 0.001,
    "eval_every_iter": 1000,
    "save_every_iter": 5000,
    "log_every_iter": 200,
    "load_experiment": None,
    "tensorboard_dir": "runs",
    "best_loss": float("inf"),
}
default_train_conf = OmegaConf.create(default_train_conf)


def compute_loss(pred, gt, criterion):
    loss = criterion(pred, gt.float())
    return loss


def val_epoch(model, val_loader, criterion, device):
    epoch_loss = 0
    with tqdm(val_loader, unit="batch") as tepoch:
        for images, gt, Ks in tepoch:
            tepoch.set_description(f"Validating ")
            #images = images.transpose(1, 2).to(device)
            data = {'view0': {'image': images[:, 0], 'depth': None, 'camera': None},
                    'view1': {'image': images[:, 1], 'depth': None, 'camera': None},
                    'K': Ks,
                    'T_0to1': Pose.from_Rt(torch.eye(3).repeat(images.shape[0], 1, 1), gt[:, :3])}
            data = batch_to_device(data, device, non_blocking=True)
            output = model(data)
            estimated_pose = output['pred_vo']    
            gt = gt.to(device)
            loss = compute_loss(estimated_pose, gt, criterion)
            epoch_loss += loss.item()
            tepoch.set_postfix(val_loss=loss.item())
            sample = {'estimated': estimated_pose[0].detach().cpu(),
                     'gt': gt[0].detach().cpu(),
                     'view0': output['view0'],
                     'view1': output['view1']}            
    return epoch_loss / len(val_loader), sample


def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, device):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1

    with tqdm(train_loader, unit="batch") as tepoch:
        for images, gt, Ks in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            #images = images.transpose(1, 2).to(device)
            
            data = {'view0': {'image': images[:, 0], 'depth': None, 'camera': None},
                    'view1': {'image': images[:, 1], 'depth': None, 'camera': None},
                    'K': Ks,
                    'T_0to1': Pose.from_Rt(torch.eye(3).repeat(images.shape[0], 1, 1), gt[:, :3])}
            data = batch_to_device(data, device, non_blocking=True)
            output = model(data)
            estimated_pose = output['pred_vo']
            gt = gt.to(device)
            loss = compute_loss(estimated_pose, gt, criterion)
            if torch.isnan(loss):
                logger.error("Encountered NaN loss!")

                logger.error(f"Images shape: {images.shape}")
                logger.error(f"GT : {gt}")
                logger.error(f"Estimated pose: {estimated_pose}")
                logger.error(f"Data keys: {list(data.keys())}")
                logger.error(f"Model output keys: {list(output.keys())}")
                
                if torch.isnan(images).any():
                    logger.error("Images contain NaN values!")
                else:
                    logger.info("Images do NOT contain NaN values.")
                raise ValueError("NaN detected during training!")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())
            tensorboard_writer.add_scalar("train/loss", loss.item(), iter)
            iter += 1
            
        origin = torch.tensor([0, 0, 0, 0, 0, 0])
        fig_cameras = draw_camera_poses([origin[3:], estimated_pose[0,3:].detach().cpu(), gt[0,3:].detach().cpu()],
                                        [origin[:3], estimated_pose[0,:3].detach().cpu(), gt[0,:3].detach().cpu()],
                                        ["origin", "estimated", "gt"], dpi=700)
        tensorboard_writer.add_figure("train/poses", fig_cameras, iter)
    return epoch_loss / len(train_loader)  


def train_tsformer(model, train_loader, val_loader, optimizer, device, config):
    criterion = pose_loss_norm
    writer = SummaryWriter(log_dir=config.train.tensorboard_dir)
    for epoch in range(config.train.epochs):
        model.train()
        if(config.vit.freeze):
            model.imgenc.eval()
        if(config.features_model.freeze):
            model.matcher.eval()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, writer, device)
        logger.info(f"Epoch {epoch}, Train loss: {train_loss}")
        writer.add_scalar("train/loss_total", train_loss, epoch)
        
        with torch.no_grad():
            model.eval()
            val_loss, sample = val_epoch(model, val_loader, criterion, device)
            logger.info(f"Epoch {epoch}, Validation loss: {val_loss}")
        
        logger.info(f"Epoch {epoch}, Validation loss: {val_loss}")
        writer.add_scalar("val/loss", val_loss, epoch)
        
        if val_loss < config.train.best_loss:
            config.train.best_loss = val_loss
            logger.info(f"New best model with loss {val_loss}. Saving checkpoint.")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": OmegaConf.to_container(config, resolve=True),
            }, os.path.join(args.experiment, "best_model.tar"))
            origin = torch.tensor([0, 0, 0, 0, 0, 0])
            fig_cameras = draw_camera_poses([origin[3:], sample['estimated'][3:], sample['gt'][3:]],
                                            [origin[:3], sample['estimated'][:3], sample['gt'][:3]],
                                            ["origin", "estimated", "gt"], dpi=700)
            img0 = sample['view0']['image'][0].detach().cpu()
            img1 = sample['view1']['image'][0].detach().cpu()
            #convert img to h,w,c
            img0 = img0.permute(1, 2, 0).numpy()*255
            img1 = img1.permute(1, 2, 0).numpy()*255
            img0 = img0.astype(np.uint8)
            img1 = img1.astype(np.uint8)
            fig_matches = draw_matches(img0, img1,
                                       sample['view0']['patches_coords'][0].detach().cpu().numpy(),
                                       sample['view1']['patches_coords'][0].detach().cpu().numpy(), 
                                       sample['view0']['scores'][0].detach().cpu().numpy())
            writer.add_figure("val/poses", fig_cameras, epoch)
            writer.add_image("val/matches", fig_matches.transpose(2, 0, 1)/255.0, epoch)
               

def get_optimizer(params, model_args):
    method = model_args["optimizer"]

    # initialize the optimizer
    if method == "Adam":
        optimizer = torch.optim.Adam(params, lr=model_args["lr"])
    elif method == "SGD":
        optimizer = torch.optim.SGD(params, lr=model_args["lr"],
                              momentum=model_args["momentum"],
                              weight_decay=model_args["weight_decay"])
    elif method == "RAdam":
        optimizer = torch.optim.RAdam(params, lr=model_args["lr"])
    elif method == "Adagrad":
        optimizer = torch.optim.Adagrad(params, lr=model_args["lr"],
                                  weight_decay=model_args["weight_decay"])

    # load checkpoint
    if model_args["checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(model_args["checkpoint_path"], model_args["checkpoint"]))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def main(args):
    conf = OmegaConf.load(args.conf)
    conf.train = OmegaConf.merge(default_train_conf, conf.train)
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    logger.info(f"Using device {device}")

    init_cp = None
    if conf.train.load_experiment:
        ckpts = sorted(glob.glob(os.path.join(conf.train.load_experiment, "checkpoint_*.tar")))
        if ckpts:
            init_cp = torch.load(ckpts[-1], map_location="cpu")
            logger.info(f"Loaded checkpoint {ckpts[-1]}")
        else:
            init_cp = torch.load(os.path.join(conf.train.load_experiment, "best_model.tar"), map_location="cpu")
            logger.info(f"Loaded checkpoint {conf.train.load_experiment}/best_model.tar")
   
    random.seed(conf.data.seed)
    np.random.seed(conf.data.seed)
    torch.manual_seed(conf.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.data.seed)

    train_loader = get_iterator(**conf.data, train=True)
    val_loader = get_iterator(**conf.data, train=False)

    os.makedirs(args.experiment, exist_ok=True)
        
    model_args = {
        "window_size": 2,  # number of frames in window
      	"pretrained_ViT": False,  # load weights from pre-trained ViT
        "checkpoint_path": "checkpoints_tsformer/",  # path to save checkpoint
        "checkpoint": None,  # checkpoint
        "optimizer": "Adam",  # optimizer [Adam, SGD, Adagrad, RAdam]
        "lr": 1e-5,  # learning rate
        "momentum": 0.9,  # SGD momentum
        "weight_decay": 1e-4,  # SGD momentum
    }
        
    model = get_photo_vo_model(conf)
    optimizer = get_optimizer(model.parameters(), model_args)
    model.to(device)

    if init_cp:
        model.load_state_dict(init_cp["model"], strict=False)
        optimizer.load_state_dict(init_cp["optimizer"])
        if conf.train.lr != optimizer.param_groups[0]['lr']:
            logger.info(f"Overriding learning rate from {optimizer.param_groups[0]['lr']} to {conf.train.lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = conf.train.lr
    
    if(conf.vit.freeze):
        logger.info("Freezing the ViT model")
        for param in model.imgenc.parameters():
            param.requires_grad = False
            
    if(conf.features_model.freeze):
        logger.info("Freezing the features model")
        for param in model.matcher.parameters():
            param.requires_grad = False 

    # check trainable weights
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name}")
        else:
            logger.info(f"Frozen parameter: {name}")
            
    logger.info(f"Training with dataset config {conf.data}")
    train_tsformer(model, train_loader, val_loader, optimizer, device, conf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--start_iter", type=int, default=0)

    args = parser.parse_args()
    main(args)