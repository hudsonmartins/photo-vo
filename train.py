import os, sys
import torch
import glob
import argparse
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from gluefactory.datasets import get_dataset
from gluefactory.utils.tensor import batch_to_device

from torch.utils.tensorboard import SummaryWriter

from utils import debug_batch
from model import get_photo_vo_model

logger = logging.getLogger("photo-vo.train")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

default_train_conf = {
    "epoch": 0,  # current epoch
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


def do_evaluation(val_loader, model, device):
    avg_losses = {}
    for it, data in enumerate(val_loader):
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            output = model(data)
            loss, output = model.loss(output)
            if torch.isnan(loss['total']).any():
                print(f"Detected NAN, skipping iteration {it}")
                continue
            avg_losses = {k: v + loss[k].item() for k, v in loss.items()}
    avg_losses = {k: v / len(val_loader) for k, v in avg_losses.items()}
    fig_matches, fig_projs, fig_patches = debug_batch(output, n_pairs=1, figs_dpi=700)
    return avg_losses, fig_matches, fig_projs, fig_patches


def train(model, train_loader, val_loader, optimizer, device, config, debug=False):
    writer = SummaryWriter(log_dir=config.train.tensorboard_dir)
    epoch = config.train.epoch
    while(epoch < config.train.epochs):
        for it, data in enumerate(train_loader):
            logger.info(f"Starting Iteration {it} in epoch {epoch}")
            tot_n_samples = (len(train_loader) * epoch + it)
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
                debug_batch(output, n_pairs=1, figs_dpi=100)
                plt.show()

            if(it % config.train.log_every_iter == 0):
                logger.info(f"[Train] Epoch {epoch} Iteration {it} Loss: {loss['total'].item()}")
                for k, v in loss.items():
                    writer.add_scalar("train/loss/" + k, v, tot_n_samples)
                fig_matches, fig_projs, fig_patches = debug_batch(output, n_pairs=1, figs_dpi=700)
                writer.add_figure("train/fig/matches", fig_matches, tot_n_samples)
                writer.add_figure("train/fig/projs", fig_projs, tot_n_samples)
                writer.add_figure("train/fig/patches", fig_patches, tot_n_samples)
                writer.add_scalar("train/epoch", epoch, tot_n_samples)

            #validation
            if(config.train.eval_every_iter > 0 and it % config.train.eval_every_iter == 0 or it == (len(train_loader) - 1)):                
                logger.info(f"[Val] Epoch: {epoch} Iteration: {it} Loss: {loss['total'].item()}")
                model.eval()
                loss, fig_matches, fig_projs, fig_patches = do_evaluation(val_loader, model, device)

                for k, v in loss.items():
                    writer.add_scalar("val/loss/" + k, v, tot_n_samples)                    
                if(fig_matches is not None):
                    writer.add_figure("val/fig/matches", fig_matches, tot_n_samples)
                if(fig_projs is not None):
                    writer.add_figure("val/fig/projs", fig_projs, tot_n_samples)
                if(fig_patches is not None):
                    writer.add_figure("val/fig/patches", fig_patches, tot_n_samples)
                writer.add_scalar("val/epoch", epoch, tot_n_samples)

                if(loss['total'].item() < config.train.best_loss):
                    best_loss = loss['total'].item()
                    config.train.best_loss = best_loss
                    config.train.epoch = epoch
                    logger.info(f"Found best model with loss {best_loss}. Saving checkpoint.")
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "conf": OmegaConf.to_container(config, resolve=True),
                    }, os.path.join(args.experiment, "best_model.tar"))

            if(config.train.save_every_iter > 0 and it % config.train.save_every_iter == 0 or it == (len(train_loader) - 1)):
                logger.info(f"Saving checkpoint at epoch {epoch} iteration {it}")
                config.train.epoch = epoch
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "conf": OmegaConf.to_container(config, resolve=True),
                }, os.path.join(args.experiment, f"checkpoint_{epoch}_{it}.tar"))
        epoch += 1

def main(args):
    conf = OmegaConf.load(args.conf)
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

    random.seed(conf.data.seed)
    torch.manual_seed(conf.data.seed)
    np.random.seed(conf.data.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(conf.data.seed)
        torch.cuda.manual_seed_all(conf.data.seed)

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

    logger.info('Training with the following configuration: ')
    logger.info(conf)

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
