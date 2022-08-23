from SorghumPartNet.models.dgcnn import DGCNNBinaryClass
from models.utils import BNMomentumScheduler, SpaceSimilarityLossV2, LeafMetrics, binary_acc
import torch.nn as nn
import pytorch_lightning as pl
from collections import namedtuple
from models.datasets import LeafDataset
from torch.utils.data import DataLoader
import torch
import torch.optim.lr_scheduler as lr_sched

class SorghumPartNetLeaf(pl.LightningModule):
        
    def __init__(self, hparams):
        '''
        Parameters
        ----------
        hparams: hyper parameters
        '''
        super(SorghumPartNetLeaf,self).__init__()

        self.hparams.update(hparams)
        self.lr_clip = 1e-5
        self.bnm_clip = 1e-2

        MyStruct = namedtuple('args', ['k', 'num_points'])
        args = MyStruct(k=15, num_points=self.hparams["num_points"])

        self.DGCNNBinaryClass = DGCNNBinaryClass(args)

        self.save_hyperparameters()
        
    def forward(self, xyz):
        
        # Normalization (min max)
        mins,_ = torch.min(xyz,axis=1)
        maxs,_ = torch.max(xyz,axis=1)
        mins = mins.unsqueeze(1)
        maxs = maxs.unsqueeze(1)
        xyz = (xyz-mins)/(maxs-mins) - 0.5

        # Classifier 
        label = self.DGCNNBinaryClass(xyz)

        return label

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams['lr_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            self.lr_clip / self.hparams['lr'],
        )
        bn_lbmd = lambda _: max(
            self.hparams['bn_momentum']
            * self.hparams['bnm_decay']
            ** (
                int(
                    self.global_step
                    * self.hparams['batch_size']
                    / self.hparams['decay_step']
                )
            ),
            self.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self,ds_path,shuff=True):
        dataset = LeafDataset(ds_path)
        loader = DataLoader(dataset, batch_size=self.hparams['batch_size'], num_workers=4, shuffle=shuff)
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['train_data'],shuff=True)

    def training_step(self, batch, batch_idx):
        points,label = batch
        
        pred_label = self(points)

        loss_fn = nn.CrossEntropyLoss()
        leaf_loss = loss_fn(pred_label, label)
        acc = binary_acc(pred_label, label)

        tensorboard_logs = {
            'train_leaf_loss': leaf_loss,
            'train_leaf_acc': acc
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': leaf_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams['val_data'],shuff=False)

    def validation_step(self, batch, batch_idx):
        points,label = batch

        pred_label = self(points)

        loss_fn = nn.CrossEntropyLoss()
        leaf_loss = loss_fn(pred_label, label)
        acc = binary_acc(pred_label, label)

        tensorboard_logs = {
            'val_leaf_loss': leaf_loss,
            'val_leaf_acc': acc
            }

        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        val_leaf_loss = torch.stack([x['val_leaf_loss'] for x in outputs]).mean()
        val_leaf_acc = torch.stack([x['val_leaf_acc'] for x in outputs]).mean()
        tensorboard_logs = {
            'val_leaf_loss': val_leaf_loss,
            'val_leaf_acc': val_leaf_acc,
            }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': val_leaf_loss, 'log': tensorboard_logs}