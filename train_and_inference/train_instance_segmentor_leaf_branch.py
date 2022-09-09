from inspect import getargs
import sys

sys.path.append("..")
from models.nn_models import *
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import argparse

hparams = {
    "batch_size": 1,
    "lc_count": 256,
    "input_channels": 0,
    "use_xyz": True,
    "lr": 0.001,
    "weight_decay": 0.0,
    "lr_decay": 0.5,
    "decay_step": 3e5,
    "bn_momentum": 0.5,
    "bnm_decay": 0.5,
    "FL_alpha": 253 / 192,
    "FL_gamma": 2,
    "feature_space_dim": 512,
    "leaf_space_threshold": 5,
    "feed_leaf_part": False,
    "feed_is_focal": False,
    "use_deep": True,
    "use_fine_leaf_index": False,
    "leaf_classifier_loss_coef": 0.0,
    "train_data": "/space/ariyanzarei/sorghum_segmentation/dataset/2022-03-10/sorghum__labeled_train.hdf5",
    "val_data": "/space/ariyanzarei/sorghum_segmentation/dataset/2022-03-10/sorghum__labeled_validation.hdf5",
    "description": "Instance segmentation of leaves - DGCNN module - MinMax normalization",
    "leaf_classifier_path": "/space/murph186/sorghum_segmentation/models/model_checkpoints/leaf_net_2022_03_10_v1/lightning_logs/version_2/checkpoints/epoch=18-step=22800.ckpt",
}


def train(debug):
    chkpt_path = "/space/ariyanzarei/sorghum_segmentation/models/model_checkpoints/SorghumPartNetInstanceWithLeafBranch"
    if not os.path.exists(chkpt_path):
        os.makedirs(chkpt_path)

    segmentor = SorghumPartNetInstanceWithLeafBranch(hparams).cuda()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_leaf_loss",
        mode="min",
    )

    if debug:
        trainer = pl.Trainer(
            accelerator="ddp",
            gpus=[0, 1],
            # gpus=1,
            max_epochs=10,
            callbacks=[checkpoint_callback],
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=chkpt_path,
            accelerator="ddp",
            gpus=[0, 1],
            # gpus=1,
            max_epochs=10,
            callbacks=[checkpoint_callback],
        )

    trainer.fit(segmentor, segmentor.train_dataloader(), segmentor.val_dataloader())


def get_args():
    parser = argparse.ArgumentParser(
        description="Sorghum 3D part segmentation prediction script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--debug",
        help="Whether to run it in the debuge mode or not",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args.debug)
