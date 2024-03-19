import torch
import cv2
import argparse
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from gluefactory import logger
from gluefactory.datasets import get_dataset
from gluefactory.utils.experiments import get_best_checkpoint, get_last_checkpoint, save_experiment
from gluefactory.utils.tensor import batch_to_device
from utils import debug_batch, get_sorted_matches
from model import get_photo_vo_model
from utils import draw_patches, draw_matches

default_train_conf = {
    "seed": "???",  # training seed
    "epochs": 1,  # number of epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "opt_regexp": None,  # regular expression to filter parameters to optimize
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "lr_schedule": {
        "type": None,  # string in {factor, exp, member of torch.optim.lr_scheduler}
        "start": 0,
        "exp_div_10": 0,
        "on_epoch": False,
        "factor": 1.0,
        "options": {},  # add lr_scheduler arguments here
    },
    "lr_scaling": [(100, ["dampingnet.const"])],
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
}
default_train_conf = OmegaConf.create(default_train_conf)


def train(model, train_loader, device, debug=False):
    model.eval()
    for it, data in enumerate(train_loader):
        data = batch_to_device(data, device, non_blocking=True)
        output = model(data)
        model.loss(output, data)
        if(debug):
           debug_batch(data, output, n_pairs=1)
           plt.show()
        
        #Compute loss
        # depth0 = data["view0"].get("depth")
        # depth1 = data["view1"].get("depth")
        # camera0, camera1 = data["view0"]["camera"], data["view1"]["camera"]
        # T_0to1, T_1to0 = data["T_0to1"], data["T_1to0"]
        # kpts0_1 = get_kpts_projection(kpts0, depth0, camera0, camera1, T_0to1)
        # kpts1_0 = get_kpts_projection(kpts1, depth1, camera1, camera0, T_1to0)
        
        

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
        epoch = init_cp["epoch"] + 1

        # get the best loss or eval metric from the previous best checkpoint
        best_cp = get_best_checkpoint(args.experiment)
        best_cp = torch.load(str(best_cp), map_location="cpu")
        best_eval = best_cp["eval"][conf.train.best_key]
        del best_cp
    else:
        # we start a new, fresh training
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = 0
        best_eval = float("inf")
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

    dataset = get_dataset(conf.data.name)(conf.data)
    train_loader = dataset.get_data_loader("train")
    print(f"Training loader has {len(train_loader)} batches")

    photo_vo_model = get_photo_vo_model(conf)
    model = photo_vo_model.matcher.to(device)
    if args.compile:
        model = torch.compile(model, mode=args.compile)
    loss_fn = model.loss
    if init_cp is not None:
        model.load_state_dict(init_cp["model"], strict=False)

    train(photo_vo_model, train_loader, device, args.debug)


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
