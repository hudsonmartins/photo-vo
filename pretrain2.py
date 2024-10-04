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

from kitti2 import get_iterator
#from model import get_photo_vo_model
from model_tsformer import build_model
from utils import batch_to_device, debug_batch_kitti, draw_camera_poses
from loss import pose_error


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

def do_evaluation(val_loader, val_size, model, device, n_images=5):
    avg_losses = None
    all_figs = {}

    for it, (images, poses) in enumerate(tqdm(val_loader, total=val_size)):
        data = {'view0': {'image': images[:, 0], 'depth': None, 'camera': None},
                'view1': {'image': images[:, 1], 'depth': None, 'camera': None},
                'T_0to1': Pose.from_Rt(torch.eye(3).repeat(images.shape[0], 1, 1), poses[:, :3])}
        data = batch_to_device(data, device)
        with torch.no_grad():
            output = model(data)
            loss, output = model.loss_kitti(output)
            if torch.isnan(loss['total']).any():
                logger.info(f"Detected NAN, skipping iteration {it}")
                continue
            avg_losses = {k: v + loss[k].mean() for k, v in loss.items()}
            if(it < n_images):
                figs = {k+'_'+str(it): v for k, v in debug_batch_kitti(output, figs_dpi=700).items()}
                all_figs = {**all_figs, **figs}

        if(it>=val_size):
            break
    if avg_losses is None:
        return None
    avg_losses = {k: v.mean() for k, v in avg_losses.items()}

    return avg_losses, all_figs


def train_tsformer(model, train_loader, val_loader, optimizer, device, config):
    writer = SummaryWriter(log_dir=config.train.tensorboard_dir)
    loss_sum = 0
    for it, (images, poses) in enumerate(tqdm(train_loader)):
        model.train()
        optimizer.zero_grad()
        images = images.to(device)
        poses = poses.to(device)

        output = model(images)
        loss = pose_error(poses, output)
        loss.mean().backward()
        optimizer.step()
        loss_sum += loss.mean().item()

        if it % config.train.log_every_iter == 0:
            logger.info(f"Iteration {it}, Loss: {loss_sum / config.train.log_every_iter}")
            writer.add_scalar("train/loss/total", loss_sum / config.train.log_every_iter, it)
            origin = torch.tensor([0, 0, 0, 0, 0, 0])
            fig_cameras = draw_camera_poses([origin, poses[0].detach().cpu().numpy(), output[0].detach().cpu().numpy()],
                                            ["origin", "gt", "pred"], dpi=700)
            writer.add_figure("train/fig/cameras", fig_cameras, it)
                
        if it % config.train.eval_every_iter == 0:
            logger.info(f"Starting validation at iteration {it}")
            model.eval()
            val_loss = 0
            for val_it, (images, poses) in enumerate(tqdm(val_loader)):
                images = images.to(device)
                poses = poses.to(device)
                output = model(images)
                loss = pose_error(poses, output)
                val_loss += loss.mean().item()
                if val_it > config.data.val_size:
                    break
            val_loss /= config.data.val_size
            logger.info(f"Validation loss at iteration {it}: {val_loss}")
            writer.add_scalar("val/loss/total", val_loss, it)
            origin = torch.tensor([0, 0, 0, 0, 0, 0])
            fig_cameras = draw_camera_poses([origin, poses[0].detach().cpu().numpy(), output[0].detach().cpu().numpy()],
                                            ["origin", "gt", "pred"], dpi=700)
            writer.add_figure("val/fig/cameras", fig_cameras, it)
            loss_sum = 0

            if val_loss < config.train.best_loss:
                best_loss = val_loss
                config.train.best_loss = best_loss
                logger.info(f"New best model with loss {val_loss}. Saving checkpoint.")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": OmegaConf.to_container(config, resolve=True),
                }, os.path.join(args.experiment, "best_model.tar"))


def train(model, train_loader, val_loader, optimizer, device, config):
    writer = SummaryWriter(log_dir=config.train.tensorboard_dir)
    for it, (images, poses) in enumerate(tqdm(train_loader)):
        model.train()
        optimizer.zero_grad()
        # Prepare data
        data = {'view0': {'image': images[:, 0], 'depth': None, 'camera': None},
                'view1': {'image': images[:, 1], 'depth': None, 'camera': None},
                'T_0to1': Pose.from_Rt(torch.eye(3).repeat(images.shape[0], 1, 1), poses[:, :3])}
        
        data = batch_to_device(data, device)
        output = model(data)
        loss, output = model.loss_kitti(output)
        if torch.isnan(loss['total']).any():
            logger.info(f"Detected NAN, skipping iteration {it}")
            continue

        loss['total'].mean().backward()
        optimizer.step()

        if it % config.train.log_every_iter == 0:
            logger.info(f"Iteration {it}, Loss: {loss['total'].mean()}")
            for k, v in loss.items():
                writer.add_scalar(f"train/loss/{k}", v.mean(), it)
            figs = debug_batch_kitti(output, figs_dpi=700)
            for k, v in figs.items():
                if(v):
                    writer.add_figure("train/fig/" + k, v, it)

        if it % config.train.eval_every_iter == 0:
            logger.info(f"Starting validation at iteration {it}")
            model.eval()
            val_loss, figs = do_evaluation(val_loader, config.data.val_size, model, device)
            if val_loss:
                logger.info(f"Validation loss at iteration {it}: {val_loss['total']}")
                for k, v in val_loss.items():
                    writer.add_scalar(f"val/loss/{k}", v, it)
                for k, v in figs.items():
                    if(v):
                        writer.add_figure("val/fig/" + k, v, it)

                if val_loss['total'] < config.train.best_loss:
                    best_loss = val_loss['total'].item()
                    config.train.best_loss = best_loss
                    logger.info(f"New best model with loss {val_loss['total']}. Saving checkpoint.")
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": OmegaConf.to_container(config, resolve=True),
                    }, os.path.join(args.experiment, "best_model.tar"))

        if config.train.save_every_iter > 0 and it % config.train.save_every_iter == 0:
            logger.info(f"Saving checkpoint at iteration {it}")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": OmegaConf.to_container(config, resolve=True),
            }, os.path.join(args.experiment, f"checkpoint_{it}.tar"))


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

    optimizer_fn = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }[conf.train.optimizer]

    random.seed(conf.data.seed)
    np.random.seed(conf.data.seed)
    torch.manual_seed(conf.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(conf.data.seed)

    train_loader = get_iterator(conf.data.path, conf.data.size, conf.data.batch_size,
                           conf.data.train_sequences, conf.data.max_skip)
    val_loader = get_iterator(conf.data.path, conf.data.size, conf.data.batch_size,
                           conf.data.val_sequences, 1)

    os.makedirs(args.experiment, exist_ok=True)
    
    
    model_args = {
        "window_size": 2,  # number of frames in window
      	"pretrained_ViT": False,  # load weights from pre-trained ViT
        "checkpoint_path": "checkpoints_tsformer/",  # path to save checkpoint
        "checkpoint": None,  # checkpoint
    }

    model_params = {
        "dim": 384,
        "image_size": (192, 640),  #(192, 640),
        "patch_size": 16,
        "attention_type": 'divided_space_time',  # ['divided_space_time', 'space_only','joint_space_time', 'time_only']
        "num_frames": model_args["window_size"],
        "num_classes": 6 * (model_args["window_size"] - 1),  # 6 DoF for each frame
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "time_only": False,
    }
    
    
    model, model_args = build_model(model_args, model_params)
    #model = get_photo_vo_model(conf)
    optimizer = optimizer_fn(model.parameters(), lr=conf.train.lr, **conf.train.optimizer_options)
    model.to(device)

    if init_cp:
        model.load_state_dict(init_cp["model"], strict=False)
        optimizer.load_state_dict(init_cp["optimizer"])
        if conf.train.lr != optimizer.param_groups[0]['lr']:
            logger.info(f"Overriding learning rate from {optimizer.param_groups[0]['lr']} to {conf.train.lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = conf.train.lr
    
    logger.info(f"Training with sequences {conf.data.train_sequences} and validation with {conf.data.val_sequences}")
    #train(model, train_loader, val_loader, optimizer, device, conf)
    train_tsformer(model, train_loader, val_loader, optimizer, device, conf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--conf", type=str, required=True)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--start_iter", type=int, default=0)

    args = parser.parse_args()
    main(args)
