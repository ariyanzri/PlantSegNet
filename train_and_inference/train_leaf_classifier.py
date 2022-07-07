from inspect import getargs
import sys
sys.path.append("..")
from models.leaf_model import *
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import argparse

hparams = {'batch_size': 4,
            'lc_count' : 256,
            'input_channels' : 0,
            'use_xyz' : True,
            'lr': 0.001,
            'weight_decay': 0.0,
            'lr_decay': 0.5,
            'decay_step': 3e5,
            'bn_momentum': 0.5,
            'bnm_decay': 0.5,
            'FL_alpha': 253/192,
            'FL_gamma': 2,
            'feature_space_dim':512,
       
            'use_deep':True,
            'use_fine_leaf_index':False,
            'train_data': '/space/murph186/sorghum_segmentation/dataset/2022-03-10_leaf_b/sorghum_leaf__labeled_train.hdf5',
            'val_data': '/space/murph186/sorghum_segmentation/dataset/2022-03-10_leaf_b/sorghum_leaf__labeled_validation.hdf5',
            'description': 'Leaf Classification (Single / Paired Leaf) - DGCNN Module - MinMax normalization'
        }

def train(debug):
        chkpt_path = '/space/murph186/sorghum_segmentation/models/model_checkpoints/SorghumPartNetLeaf'
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)

        classifier = SorghumPartNetLeaf(hparams).cuda()

        checkpoint_callback = ModelCheckpoint(
            save_top_k=3,
            monitor='val_leaf_loss',
            mode='min',
            )
    
        if debug:
            trainer = pl.Trainer(
                # default_root_dir=chkpt_path,
                accelerator="ddp",gpus=[1],
                # gpus=1,
                max_epochs=20,
                callbacks=[checkpoint_callback],
            )
        else:
            trainer = pl.Trainer(
                default_root_dir=chkpt_path,
                accelerator="ddp",gpus=[1],
                # gpus=1,
                max_epochs=20,
                callbacks=[checkpoint_callback],
            )

        trainer.fit(classifier, classifier.train_dataloader(), classifier.val_dataloader())

def get_args():
    parser = argparse.ArgumentParser(
        description='Sorghum 3D single / paired leaf classification script.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d',
                        '--debug',
                        help='Whether to run it in the debug mode or not',
                        action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args.debug)