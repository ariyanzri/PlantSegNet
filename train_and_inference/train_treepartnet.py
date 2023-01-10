import pytorch_lightning as pl
import argparse
import json
import os
import sys

sys.path.append("..")
from models.nn_models import *

from pytorch_lightning.callbacks import ModelCheckpoint


def get_args():
    parser = argparse.ArgumentParser(
        description="TreePartNet Instance segmentation training script.",
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
        help="The path to the directory in which the model checkpoints will be saved. ",
        metavar="output",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-d",
        "--debug",
        help="Whether to run it in the debuge mode or not",
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

    chkpt_path = os.path.join(args.output, hparams["dataset"], "TreePartNet")

    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    if args.debug:
        trainer = pl.Trainer(
            accelerator="ddp",
            gpus=hparams["gpus"],
            # gpus=1,
            max_epochs=hparams["epochs"],
            callbacks=[checkpoint_callback],
        )
        segmentor = TreePartNet(hparams, True).cuda()
    else:
        trainer = pl.Trainer(
            default_root_dir=chkpt_path,
            accelerator="ddp",
            gpus=hparams["gpus"],
            # gpus=1,
            max_epochs=hparams["epochs"],
            callbacks=[checkpoint_callback],
        )
        segmentor = TreePartNet(hparams, False).cuda()

    trainer.fit(segmentor, segmentor.train_dataloader(), segmentor.val_dataloader())


if __name__ == "__main__":
    train()
