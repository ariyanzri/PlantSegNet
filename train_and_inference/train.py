import pytorch_lightning as pl
import argparse
import json
import os
import shutil
import sys

sys.path.append("..")
from models.nn_models import *

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def get_args():
    parser = argparse.ArgumentParser(
        description="Generalized training script for all the experiments and models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--hparam",
        help="The path to the hyperparameters json file. ",
        metavar="hparam",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="The path to the results directory. Subdirectories for dataset and model will be automatically created under this directory. ",
        metavar="output",
        required=False,
        default="/speedy/ariyanzarei/sorghum_segmentation/results/training_logs",
        type=str,
    )

    parser.add_argument(
        "-d",
        "--debug",
        help="Whether to run it in the debuge mode or not",
        action="store_true",
    )

    parser.add_argument(
        "-f",
        "--force",
        help="Whether to forcefully overwrite the experiment if exist or not. ",
        action="store_true",
    )

    return parser.parse_args()


def get_hparam(path):
    with open(path, "r") as f:
        hparams = json.load(f)
    return hparams


def train():
    args = get_args()
    hparams = get_hparam(args.hparam)
    version_name = hparams["experiment_id"]
    if args.debug:
        version_name += "_debug"

    chkpt_path = os.path.join(args.output, hparams["model_name"], hparams["dataset"])

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    if os.path.exists(os.path.join(chkpt_path, version_name)):
        if args.force:
            shutil.rmtree(os.path.join(chkpt_path, version_name))
        else:
            print(
                ":: There is a folder for this version of experiments. Please use -f to overwrite this experiment."
            )
            return

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=hparams["early_stopping_metric"],
        mode=hparams["early_stopping_mode"],
    )

    tensorboard_callback = TensorBoardLogger(
        save_dir=chkpt_path, name="", default_hp_metric=False, version=version_name
    )

    early_stopping_callback = EarlyStopping(
        monitor=hparams["early_stopping_metric"],
        mode=hparams["early_stopping_mode"],
        patience=hparams["early_stopping_patience"],
    )

    ModelClass = eval(hparams["model_name"])

    trainer = pl.Trainer(
        default_root_dir=chkpt_path,
        accelerator="ddp",
        gpus=hparams["gpus"],
        max_epochs=hparams["epochs"],
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=tensorboard_callback,
    )

    model = ModelClass(hparams, args.debug).cuda()

    trainer.fit(model, model.train_dataloader(), model.val_dataloader())


if __name__ == "__main__":
    train()
