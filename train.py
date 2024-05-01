import os, sys
import torch
import glob
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from gluefactory import logger
from gluefactory.datasets import get_dataset
from gluefactory.utils.tensor import batch_to_device

from torch.utils.tensorboard import SummaryWriter

from utils import debug_batch, get_sorted_matches
from model import get_photo_vo_model

default_train_conf = {
    "seed": 42,  # random seed
    "epochs": 1,  # number of epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "eval_every_iter": 1000,  # interval for evaluation on the validation set
    "save_every_iter": 5000,  # interval for saving the current checkpoint
    "log_every_iter": 200,  # interval for logging the loss to the console
    "load_experiment": None,  # initialize the model from a previous experiment
    "tensorboard_dir": "runs",  # directory for tensorboard logs
    "best_loss": float("inf"),  # best loss for the model
}
default_train_conf = OmegaConf.create(default_train_conf)

def train(model, train_loader, val_loader, optimizer, device, config, debug=False):
    writer = SummaryWriter(log_dir=config.train.tensorboard_dir)

    for it, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        data = batch_to_device(data, device, non_blocking=True)
        output = model(data)
        loss, output = model.loss(output)
        if torch.isnan(loss['total']).any():
            print(f"Detected NAN, skipping iteration {it}")
            continue
        loss['total'].backward()
        optimizer.step()

        if(debug):
           debug_batch(output, n_pairs=1)
           plt.show()

        if(it % config.train.log_every_iter == 0):
            logger.info(f"[Train] Iteration {it} Loss: {loss['total'].item()}")
            writer.add_scalar("train/loss/total", loss['total'].item(), it)
            writer.add_scalar("train/loss/photometric", loss['photometric_loss'].item(), it)
            writer.add_scalar("train/loss/pose", loss['pose_error'].item(), it)
            writer.add_scalar("train/loss/match", loss['match_loss'].item(), it)
            fig_matches, fig_projs, fig_patches = debug_batch(output, n_pairs=1)
            writer.add_figure("train/fig/matches", fig_matches, it)
            writer.add_figure("train/fig/projs", fig_projs, it)
            writer.add_figure("train/fig/patches", fig_patches, it)

        #validation
        if(it % config.train.eval_every_iter == 0 or it == (len(train_loader) - 1)):
            model.eval()
            avg_losses = {k: 0 for k in loss.keys()}
            output = None
            for it, data in enumerate(val_loader):
                data = batch_to_device(data, device, non_blocking=True)
                with torch.no_grad():
                    output = model(data)
                    loss, output = model.loss(output)
                    if torch.isnan(loss['total']).any():
                        print(f"Detected NAN, skipping iteration {it}")
                        continue
                    avg_losses = {k: v + loss[k].item() for k, v in avg_losses.items()}

            avg_losses = {k: v / len(val_loader) for k, v in avg_losses.items()}
            
            if(output is not None):
                logger.info(f"[Val] Loss: {avg_losses['total']}")
                writer.add_scalar("val/loss/total", avg_losses['total'], it)
                writer.add_scalar("val/loss/photometric", avg_losses['photometric_loss'], it)
                writer.add_scalar("val/loss/pose", avg_losses['pose_error'], it)
                writer.add_scalar("val/loss/match", avg_losses['match_loss'], it)
                fig_matches, fig_projs, fig_patches = debug_batch(output, n_pairs=3)
                writer.add_figure("val/fig/matches", fig_matches, it)
                writer.add_figure("val/fig/projs", fig_projs, it)
                writer.add_figure("val/fig/patches", fig_patches, it)
            
                if(avg_losses['total'] < config.train.best_loss):
                        best_loss = avg_losses['total']
                        config.train.best_loss = best_loss
                        logger.info(f"Found best model with loss {best_loss}. Saving checkpoint.")
                        torch.save({
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "conf": OmegaConf.to_container(config, resolve=True),
                            "epoch": it,
                        }, os.path.join(args.experiment, "best_model.tar"))

        if(it % config.train.save_every_iter == 0 or it == (len(train_loader) - 1)):
            logger.info(f"Saving checkpoint at iteration {it}")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "conf": OmegaConf.to_container(config, resolve=True),
                "epoch": it,
            }, os.path.join(args.experiment, f"checkpoint_{it}.tar"))


def main(args):
    logger.info('Training with the following configuration: ')
    conf = OmegaConf.load(args.conf)
    logger.info(conf.data)
    assert conf.features_model.name == 'two_view_pipeline'
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    logger.info(f"Using device {device}")
    
    if args.restore:
        logger.info(f"Restoring from previous training of {args.experiment}")
        ckpts = glob.glob(os.path.join(args.experiment, "checkpoint_*.tar"))
        ckpts = [os.path.basename(ckpt) for ckpt in ckpts]
        if len(ckpts) > 0:
            init_cp_name = sorted(ckpts)[-1]
            init_cp = torch.load(os.path.join(args.experiment, init_cp_name), map_location="cpu")
            logger.info(f"Will load model {init_cp_name}")
        else:
            init_cp = torch.load(os.path.join(args.experiment, "best_model.tar"), map_location="cpu")
            logger.info(f"Will load model best_model.tar")
        
        conf = OmegaConf.merge(OmegaConf.create(init_cp["conf"]), conf)
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
    else:
        # we start a new, fresh training
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        if conf.train.load_experiment:
            logger.info(f"Will fine-tune from weights of {conf.train.load_experiment}")
            ckpts = glob.glob(os.path.join(args.experiment, "checkpoint_*.tar"))
            ckpts = [os.path.basename(ckpt) for ckpt in ckpts]
            if len(ckpts) > 0:
                init_cp_name = sorted(ckpts)[-1]
                init_cp = torch.load(os.path.join(args.experiment, init_cp_name), map_location="cpu")
                logger.info(f"Will load model {init_cp_name}")
            else:
                init_cp = torch.load(os.path.join(args.experiment, "best_model.tar"), map_location="cpu")
                logger.info(f"Will load model best_model.tar")


            # load the model config of the old setup, and overwrite with current config
            conf.photo_vo = OmegaConf.merge(
                OmegaConf.create(init_cp["conf"]).photo_vo, conf.photo_vo
            )
            conf.features_model = OmegaConf.merge(
                OmegaConf.create(init_cp["conf"]).features_model, conf.features_model
            )
        else:
            init_cp = None

    optimizer_fn = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }[conf.train.optimizer]

    random.seed(conf.train.seed)
    torch.manual_seed(conf.train.seed)
    np.random.seed(conf.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(conf.train.seed)
        torch.cuda.manual_seed_all(conf.train.seed)

    dataset = get_dataset(conf.data.name)(conf.data)
    train_loader = dataset.get_data_loader("train")
    val_loader = dataset.get_data_loader("val")
    logger.info(f"Training with {len(train_loader)} batches and validating with {len(val_loader)} batches")

    os.makedirs(args.experiment, exist_ok=True)
    photo_vo_model = get_photo_vo_model(conf)
    photo_vo_model.to(device)

    if init_cp is not None:
        photo_vo_model.load_state_dict(init_cp["model"], strict=False)
        logger.info(f"Loaded model from {args.experiment}")
    
    optimizer = optimizer_fn(
        photo_vo_model.parameters(), lr=conf.train.lr, **conf.train.optimizer_options
    )
    if args.restore:
        optimizer.load_state_dict(init_cp["optimizer"])

    train(photo_vo_model, train_loader, val_loader, optimizer, device, conf, args.debug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--conf", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument(
        "--compile",
        default=None,
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
    )

    args = parser.parse_args()
    main(args)
