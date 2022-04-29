import sys
sys.path.append("..")
from models.nn_models import *
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

hparams = {'batch_size': 1,
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
            'leaf_space_threshold':5,
            'feed_leaf_part':False,
            'feed_is_focal':False,
            'use_deep':True,
            'use_fine_leaf_index':False,
            'train_data': '/space/ariyanzarei/sorghum_segmentation/dataset/2022-03-10/sorghum__labeled_train.hdf5',
            'val_data': '/space/ariyanzarei/sorghum_segmentation/dataset/2022-03-10/sorghum__labeled_validation.hdf5',
            'description': 'DGCNN module for taking points into hidden space. Use leaf indexes. Modified the loss function to use negative valus and clamp (Idea from <A Convolutional Neural Network for Point Cloud Instance Segmentation in Cluttered Scene Trained by Synthetic Data without Color>). Separate Pointnet module for soil segmentation and focal non-focal plant classification (multi-class). Normalizing points in the beginning.'
        }

def train():
        chkpt_path = '/space/ariyanzarei/sorghum_segmentation/models/model_checkpoints/SorghumPartNetSemantic'
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)

        segmentor = SorghumPartNetSemantic(hparams).cuda()

        checkpoint_callback = ModelCheckpoint(
            save_top_k=3,
            monitor='val_leaf_loss',
            mode='min',
            )
    
        trainer = pl.Trainer(
            default_root_dir=chkpt_path,
            # accelerator="ddp",gpus=[0,1],
            gpus=1,
            max_epochs=10,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(segmentor, segmentor.train_dataloader(), segmentor.val_dataloader())

train()