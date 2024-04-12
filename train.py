import torch
import tqdm
import argparse
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from gluefactory import logger
from gluefactory.datasets import get_dataset
from gluefactory.utils.tensor import batch_to_device
from gluefactory.utils.experiments import get_last_checkpoint, get_best_checkpoint
from torch.utils.tensorboard import SummaryWriter

from utils import debug_batch, get_sorted_matches
from model import get_photo_vo_model

default_train_conf = {
    "seed": "???",  # training seed
    "epochs": 1,  # number of epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "opt_regexp": None,  # regular expression to filter parameters to optimize
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "eval_every_iter": 1000,  # interval for evaluation on the validation set
    "save_every_iter": 5000,  # interval for saving the current checkpoint
    "log_every_iter": 200,  # interval for logging the loss to the console
    "log_grad_every_iter": None,  # interval for logging gradient hists
    "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
    "keep_last_checkpoints": 10,  # keep only the last X checkpoints
    "load_experiment": None,  # initialize the model from a previous experiment
    "median_metrics": [],  # add the median of some metrics
    "recall_metrics": {},  # add the recall of some metrics
    "pr_metrics": {},  # add pr curves, set labels/predictions/mask keys
    "best_key": "loss/total",  # key to use to select the best checkpoint
    "dataset_callback_fn": None,  # data func called at the start of each epoch
    "dataset_callback_on_val": False,  # call data func on val data?
    "clip_grad": None,
    "pr_curves": {},
    "plot": None,
    "submodules": [],
    "tensorboard_dir": "runs",  # directory for tensorboard logs
}
default_train_conf = OmegaConf.create(default_train_conf)


def train(model, train_loader, val_loader, optimizer, device, config, debug=False):
    writer = SummaryWriter(log_dir=config.train.tensorboard_dir)
    best_loss = float("inf")
    for it, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        data = batch_to_device(data, device, non_blocking=True)
        output = model(data)
        loss = model.loss(output)
        if torch.isnan(loss['total']).any():
            print(f"Detected NAN, skipping iteration {it}")
            del output
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
            fig = debug_batch(output, n_pairs=1)
            writer.add_figure("train/fig/debug", fig, it)

        #validation            
        if(it % config.train.eval_every_iter == 0 or it == (len(train_loader) - 1)):
            model.eval()
            avg_losses = {k: 0 for k in loss.keys()}
            #for it, data in enumerate(tqdm(val_loader, desc="Validation", ascii=True)):
            for it, data in enumerate(val_loader):
                data = batch_to_device(data, device, non_blocking=True)
                with torch.no_grad():
                    output = model(data)
                    loss = model.loss(output)
                    avg_losses = {k: v + loss[k].item() for k, v in avg_losses.items()}
                    if torch.isnan(loss['total']).any():
                        print(f"Detected NAN, skipping iteration {it}")
                        del output
                        continue
            avg_losses = {k: v / len(val_loader) for k, v in avg_losses.items()}
            

            logger.info(f"[Val] Loss: {avg_losses['total']}")
            writer.add_scalar("val/loss/total", avg_losses['total'], it)
            writer.add_scalar("val/loss/photometric", avg_losses['photometric_loss'], it)
            writer.add_scalar("val/loss/pose", avg_losses['pose_error'], it)
            writer.add_scalar("val/loss/match", avg_losses['match_loss'], it)
            fig = debug_batch(output, n_pairs=3)
            writer.add_figure("val/fig/debug", fig, it)        
        
            if(avg_losses['total'] < best_loss):
                    best_loss = avg_losses['total']
                    logger.info(f"Found best model with loss {best_loss}. Saving checkpoint.")
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "conf": OmegaConf.to_container(config, resolve=True),
                        "epoch": it,
                    }, f"checkpoint_best.tar") 

        if(it % config.train.save_every_iter == 0 or it == (len(train_loader) - 1)):
            logger.info(f"Saving checkpoint at iteration {it}")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "conf": OmegaConf.to_container(config, resolve=True),
                "epoch": it,
            }, f"checkpoint_{it}.tar")


def main(args):
    logger.info('Training with the following configuration: ')
    conf = OmegaConf.load(args.conf)
    logger.info(conf.data)
    assert conf.features_model.name == 'two_view_pipeline'
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
    logger.info(f"Using device {device}")
    
    if args.restore:
        logger.info(f"Restoring from previous training of {args.experiment}")
        try:
            init_cp = get_last_checkpoint(args.experiment, allow_interrupted=False)
        except AssertionError:
            init_cp = get_best_checkpoint(args.experiment)
        logger.info(f"Restoring from checkpoint {init_cp.name}")
        init_cp = torch.load(str(init_cp), map_location="cpu")
        conf = OmegaConf.merge(OmegaConf.create(init_cp["conf"]), conf)
        conf.train = OmegaConf.merge(default_train_conf, conf.train)

        # get the best loss or eval metric from the previous best checkpoint
        best_cp = get_best_checkpoint(args.experiment)
        best_cp = torch.load(str(best_cp), map_location="cpu")
        del best_cp
    else:
        # we start a new, fresh training
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        if conf.train.load_experiment:
            logger.info(f"Will fine-tune from weights of {conf.train.load_experiment}")
            # the user has to make sure that the weights are compatible
            try:
                init_cp = get_last_checkpoint(conf.train.load_experiment)
            except AssertionError:
                init_cp = get_best_checkpoint(conf.train.load_experiment)
            # init_cp = get_last_checkpoint(conf.train.load_experiment)
            init_cp = torch.load(str(init_cp), map_location="cpu")
            # load the model config of the old setup, and overwrite with current config
            conf.model = OmegaConf.merge(
                OmegaConf.create(init_cp["conf"]).model, conf.model
            )
        else:
            init_cp = None

    optimizer_fn = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }[conf.train.optimizer]

    dataset = get_dataset(conf.data.name)(conf.data)
    train_loader = dataset.get_data_loader("train")
    val_loader = dataset.get_data_loader("val")
    print(f"Training loader has {len(train_loader)} batches")
    print(f"Validation loader has {len(val_loader)} batches")

    photo_vo_model = get_photo_vo_model(conf)
    photo_vo_model.matcher.to(device)
    if args.compile:
        photo_vo_model.matcher = torch.compile(photo_vo_model.matcher, mode=args.compile)
    if init_cp is not None:
        photo_vo_model.load_state_dict(init_cp["model"], strict=False)
    
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
