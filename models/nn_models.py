from models.dgcnn import DGCNNFeatureSpace, DGCNNSemanticSegmentor
from models.extensions import LeafBranch
import torch
import numpy as np
import os
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.datasets import (
    SorghumDataset,
    SorghumDatasetWithNormals,
    TreePartNetDataset,
    TreePartNetOriginalDataset,
    PartNetDataset,
    SorghumDatasetTPNFormat,
)
from models.dgcnn import *
from models.dgcnn_new import DGCNN_partseg
from collections import namedtuple
from models.utils import (
    BNMomentumScheduler,
    SpaceSimilarityLossV2,
    SpaceSimilarityLossV3,
    SpaceSimilarityLossV4,
    LeafMetrics,
    SemanticMetrics,
    LeafMetricsTraining,
)
from models.treepartnet_utils import (
    PointnetSAModuleMSG,
    PointnetFPModule,
    build_shared_mlp,
    ScaledDot,
    FocalLoss,
)
from models.leaf_model import SorghumPartNetLeaf
from data.load_raw_data import load_real_ply_with_labels
import matplotlib.pyplot as plt
import torchvision
from sklearn.cluster import DBSCAN
from data.utils import distinct_colors


class SorghumPartNetSemantic(pl.LightningModule):
    def __init__(self, hparams, debug=False):
        """
        Parameters
        ----------
        hparams: hyper parameters
        """
        super(SorghumPartNetSemantic, self).__init__()

        self.is_debug = debug
        self.hparams.update(hparams)
        self.lr_clip = 1e-5
        self.bnm_clip = 1e-2

        # if "emb_dims" in self.hparams:
        #     MyStruct = namedtuple("args", ["k", "emb_dims", "dropout"])
        #     args = MyStruct(
        #         k=self.hparams["dgcnn_k"],
        #         emb_dims=self.hparams["emb_dims"],
        #         dropout=self.hparams["dropout"],
        #     )

        #     self.DGCNN_semantic_segmentor = DGCNN_partseg(args, 3).double()
        # else:
        #     self.DGCNN_semantic_segmentor = DGCNNSemanticSegmentor(
        #         self.hparams["dgcnn_k"],
        #         input_dim=(
        #             3 if "input_dim" not in self.hparams else self.hparams["input_dim"]
        #         ),
        #     ).double()

        self.DGCNN_semantic_segmentor = DGCNNSemanticSegmentor(
            self.hparams["dgcnn_k"],
            input_dim=(
                3 if "input_dim" not in self.hparams else self.hparams["input_dim"]
            ),
        ).double()

        self.save_hyperparameters()

    def forward(self, xyz):

        # Normalization
        if (
            "normalization" not in self.hparams
            or self.hparams["normalization"] == "min-max"
        ):
            mins, _ = torch.min(xyz, axis=1)
            maxs, _ = torch.max(xyz, axis=1)
            mins = mins.unsqueeze(1)
            maxs = maxs.unsqueeze(1)
            xyz = (xyz - mins) / (maxs - mins) - 0.5
        elif self.hparams["normalization"] == "mean-std":
            mean = torch.mean(xyz, axis=1)
            mean = mean.unsqueeze(1).repeat(1, xyz.shape[1], 1)
            std = torch.std(xyz, axis=1)
            std = std.unsqueeze(1).repeat(1, xyz.shape[1], 1)
            xyz = (xyz - mean) / std

        # Semantic Label Prediction
        semantic_label_pred = self.DGCNN_semantic_segmentor(xyz)

        return semantic_label_pred

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.lr_clip / self.hparams["lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["bn_momentum"]
            * self.hparams["bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self, ds_path, shuff=True):
        if "use_normals" not in self.hparams:
            dataset = SorghumDataset(ds_path)
        else:
            dataset = SorghumDatasetWithNormals(
                ds_path,
                self.hparams["use_normals"],
                self.hparams["std_noise"],
                self.hparams["duplicate_ground_prob"],
                self.hparams["focal_only_prob"],
                debug=self.is_debug,
            )

        loader = DataLoader(
            dataset, batch_size=self.hparams["batch_size"], num_workers=4, shuffle=shuff
        )
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["train_data"], shuff=True)

    def training_step(self, batch, batch_idx):
        if "use_normals" not in self.hparams:
            points, _, semantic_label, _, _ = batch
        else:
            points, semantic_label = batch

        pred_semantic_label = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        semantic_label_loss = critirion(pred_semantic_label, semantic_label)

        with torch.no_grad():
            # semantic_label_acc = (
            #     (torch.argmax(pred_semantic_label, dim=1) == semantic_label)
            #     .float()
            #     .mean()
            # )
            metric_calculator = SemanticMetrics()
            semantic_label_acc = metric_calculator(
                torch.argmax(pred_semantic_label, dim=1), semantic_label
            )

        tensorboard_logs = {
            "train_semantic_label_loss": semantic_label_loss,
            "train_semantic_label_acc": semantic_label_acc,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {"loss": semantic_label_loss, "log": tensorboard_logs}

    # def log_pointcloud_image(self,)
    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["val_data"], shuff=False)

    def validation_step(self, batch, batch_idx):
        if "use_normals" not in self.hparams:
            points, _, semantic_label, _, _ = batch
        else:
            points, semantic_label = batch

        pred_semantic_label = self(points)

        critirion = torch.nn.CrossEntropyLoss()
        semantic_label_loss = critirion(pred_semantic_label, semantic_label)

        # semantic_label_acc = (
        #     (torch.argmax(pred_semantic_label, dim=1) == semantic_label).float().mean()
        # )
        metric_calculator = SemanticMetrics()
        semantic_label_acc = metric_calculator(
            torch.argmax(pred_semantic_label, dim=1), semantic_label
        )

        tensorboard_logs = {
            "val_semantic_label_loss": semantic_label_loss,
            "val_semantic_label_acc": semantic_label_acc,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return tensorboard_logs

    def validation_epoch_end(self, batch):
        self.validation_real_data()

    def validation_real_data(self):
        real_data_path = self.hparams["real_data"]

        device_name = "cpu"
        device = torch.device(device_name)

        semantic_model = self.to(device)
        semantic_model.DGCNN_semantic_segmentor.device = device_name

        files = os.listdir(real_data_path)
        accs = []
        pred_images = []

        for file in files:
            path = os.path.join(real_data_path, file)
            points, _, semantic_labels = load_real_ply_with_labels(path)
            points = torch.tensor(points, dtype=torch.float64).to(device)
            if (
                "use_normals" in semantic_model.hparams
                and semantic_model.hparams["use_normals"]
            ):
                pred_semantic_label = semantic_model(
                    torch.unsqueeze(points, dim=0).to(device)
                )
            else:
                pred_semantic_label = semantic_model(
                    torch.unsqueeze(points[:, :3], dim=0).to(device)
                )

            pred_semantic_label = F.softmax(pred_semantic_label, dim=1)
            pred_semantic_label = pred_semantic_label[0].cpu().detach().numpy().T
            pred_semantic_label_labels = np.argmax(pred_semantic_label, 1)

            colors = np.column_stack(
                (
                    pred_semantic_label_labels,
                    pred_semantic_label_labels,
                    pred_semantic_label_labels,
                )
            ).astype("float32")
            colors[colors[:, 0] == 0, :] = [0.3, 0.1, 0]
            colors[colors[:, 0] == 1, :] = [0, 0.7, 0]
            colors[colors[:, 0] == 2, :] = [0, 0, 0.7]

            metric_calculator = SemanticMetrics()
            acc = metric_calculator(
                torch.tensor(pred_semantic_label_labels), torch.tensor(semantic_labels)
            )

            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=2, c=colors)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(f"Accuracy: {acc:.2f}")
            fig.canvas.draw()
            X = (
                torch.tensor(np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3])
                .transpose(0, 2)
                .transpose(1, 2)
            )
            X = torchvision.transforms.functional.resize(X, (1000, 1000))
            fig.close()
            accs.append(acc)
            pred_images.append(X)

        accs = torch.tensor(accs)
        grid = torch.cat(pred_images, 1)
        self.logger.experiment.add_image(
            "pred_real_data", grid, self.trainer.current_epoch
        )

        # self.log(
        #     "test_real_acc",
        #     torch.mean(accs),
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=False,
        #     logger=True,
        # )
        self.logger.experiment.add_scalar(
            "test_real_acc", torch.mean(accs), self.trainer.current_epoch
        )

        semantic_model = self.to(torch.device("cuda"))
        semantic_model.DGCNN_semantic_segmentor.device = "cuda"


class SorghumPartNetInstance(pl.LightningModule):
    def __init__(self, hparams, debug=False):
        """
        Parameters
        ----------
        hparams: hyper parameters
        """
        super(SorghumPartNetInstance, self).__init__()

        self.is_debug = debug
        self.hparams.update(hparams)
        self.lr_clip = 1e-5
        self.bnm_clip = 1e-2

        MyStruct = namedtuple("args", "k")
        if "dgcnn_k" in self.hparams:
            args = MyStruct(k=self.hparams["dgcnn_k"])
        else:
            args = MyStruct(k=15)

        self.DGCNN_feature_space = DGCNNFeatureSpace(
            args, (3 if "input_dim" not in self.hparams else self.hparams["input_dim"])
        ).double()

        self.save_hyperparameters()

    def forward(self, xyz):

        # Normalization
        if (
            "normalization" not in self.hparams
            or self.hparams["normalization"] == "min-max"
        ):
            mins, _ = torch.min(xyz, axis=1)
            maxs, _ = torch.max(xyz, axis=1)
            mins = mins.unsqueeze(1)
            maxs = maxs.unsqueeze(1)
            xyz = (xyz - mins) / (maxs - mins) - 0.5
        if (
            "normalization" not in self.hparams
            or self.hparams["normalization"] == "mean-std"
        ):
            mean = torch.mean(xyz, axis=1)
            mean = mean.unsqueeze(1).repeat(1, xyz.shape[1], 1)
            std = torch.std(xyz, axis=1)
            std = std.unsqueeze(1).repeat(1, xyz.shape[1], 1)
            xyz = (xyz - mean) / std

        # Instance
        dgcnn_features = self.DGCNN_feature_space(xyz)

        return dgcnn_features

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.lr_clip / self.hparams["lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["bn_momentum"]
            * self.hparams["bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self, ds_path, shuff=True):
        if "use_normals" not in self.hparams:
            dataset = SorghumDataset(ds_path)
        else:
            if "dataset" not in self.hparams or self.hparams["dataset"] == "SPN":
                dataset = SorghumDatasetWithNormals(
                    ds_path,
                    self.hparams["use_normals"],
                    self.hparams["std_noise"],
                    debug=self.is_debug,
                )
            elif self.hparams["dataset"] == "TPN":
                dataset = TreePartNetDataset(
                    ds_path,
                    debug=self.is_debug,
                )
            elif self.hparams["dataset"] == "PN":
                dataset = PartNetDataset(
                    ds_path,
                    debug=self.is_debug,
                )

        loader = DataLoader(
            dataset, batch_size=self.hparams["batch_size"], num_workers=4, shuffle=shuff
        )
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["train_data"], shuff=True)

    def training_step(self, batch, batch_idx):
        if "use_normals" not in self.hparams:
            points, _, _, _, leaf = batch
        else:
            points, leaf = batch

        pred_leaf_features = self(points)

        if "loss_fn" not in self.hparams:
            criterion_cluster = SpaceSimilarityLossV2()
        elif self.hparams["loss_fn"] == "v3":
            criterion_cluster = SpaceSimilarityLossV3(points)
        elif self.hparams["loss_fn"] == "v4":
            criterion_cluster = SpaceSimilarityLossV4(points)

        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        leaf_metrics = LeafMetricsTraining(self.hparams["leaf_space_threshold"])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features, leaf)

        tensorboard_logs = {
            "train_leaf_loss": leaf_loss,
            "train_leaf_accuracy": Acc,
            "train_leaf_precision": Prec,
            "train_leaf_recall": Rec,
            "train_leaf_f1": F,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {"loss": leaf_loss, "log": tensorboard_logs}

    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["val_data"], shuff=False)

    def validation_step(self, batch, batch_idx):
        if "use_normals" not in self.hparams:
            points, _, _, _, leaf = batch
        else:
            points, leaf = batch

        pred_leaf_features = self(points)

        if "loss_fn" not in self.hparams:
            criterion_cluster = SpaceSimilarityLossV2()
        elif self.hparams["loss_fn"] == "v3":
            criterion_cluster = SpaceSimilarityLossV3(points)
        elif self.hparams["loss_fn"] == "v4":
            criterion_cluster = SpaceSimilarityLossV4(points)

        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        leaf_metrics = LeafMetricsTraining(self.hparams["leaf_space_threshold"])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features, leaf)

        tensorboard_logs = {
            "val_leaf_loss": leaf_loss,
            "val_leaf_accuracy": Acc,
            "val_leaf_precision": Prec,
            "val_leaf_recall": Rec,
            "val_leaf_f1": F,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return tensorboard_logs

    def validation_epoch_end(self, batch):
        if "real_data" in self.hparams:
            self.validation_real_data()

    def validation_real_data(self):
        real_data_path = self.hparams["real_data"]

        device_name = "cpu"
        device = torch.device(device_name)

        instance_model = self.to(device)
        instance_model.DGCNN_feature_space.device = device_name

        files = os.listdir(real_data_path)
        accs = []
        precisions = []
        recals = []
        f1s = []
        pred_images = []

        for file in files:
            path = os.path.join(real_data_path, file)
            main_points, instance_labels, semantic_labels = load_real_ply_with_labels(
                path
            )
            points = main_points[semantic_labels == 1]
            instance_labels = instance_labels[semantic_labels == 1]

            points = torch.tensor(points, dtype=torch.float64).to(device)
            if (
                "use_normals" in instance_model.hparams
                and instance_model.hparams["use_normals"]
            ):
                pred_instance_features = instance_model(
                    torch.unsqueeze(points, dim=0).to(device)
                )
            else:
                pred_instance_features = instance_model(
                    torch.unsqueeze(points[:, :3], dim=0).to(device)
                )

            pred_instance_features = (
                pred_instance_features.cpu().detach().numpy().squeeze()
            )
            clustering = DBSCAN(eps=1, min_samples=10).fit(pred_instance_features)
            pred_final_cluster = clustering.labels_

            d_colors = distinct_colors(len(list(set(pred_final_cluster))))
            colors = np.zeros((pred_final_cluster.shape[0], 3))
            for i, l in enumerate(list(set(pred_final_cluster))):
                colors[pred_final_cluster == l, :] = d_colors[i]

            non_focal_points = main_points[semantic_labels == 2]
            ground_points = main_points[semantic_labels == 0]

            non_focal_color = [0, 0, 0.7, 0.3]
            ground_color = [0.3, 0.1, 0, 0.3]

            metric_calculator = LeafMetrics()
            acc, precison, recal, f1 = metric_calculator(
                torch.tensor(pred_final_cluster).unsqueeze(0).unsqueeze(-1),
                torch.tensor(instance_labels).unsqueeze(0).unsqueeze(-1),
            )

            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, c=colors)
            ax.scatter(
                non_focal_points[:, 0],
                non_focal_points[:, 1],
                non_focal_points[:, 2],
                s=1,
                color=non_focal_color,
            )
            ax.scatter(
                ground_points[:, 0],
                ground_points[:, 1],
                ground_points[:, 2],
                s=1,
                color=ground_color,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(
                f"acc: {acc*100:.2f} - precision: {precison:.2f} - recall: {recal:.2f} - f1: {f1:.2f}"
            )
            fig.canvas.draw()
            X = (
                torch.tensor(np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3])
                .transpose(0, 2)
                .transpose(1, 2)
            )
            X = torchvision.transforms.functional.resize(X, (1000, 1000))
            fig.close()
            accs.append(acc)
            precisions.append(precison)
            recals.append(recal)
            f1s.append(f1)
            pred_images.append(X)

        accs = torch.tensor(accs)
        precisions = torch.tensor(precisions)
        recals = torch.tensor(recals)
        f1s = torch.tensor(f1s)

        tensorboard_logs = {
            "test_real_acc": torch.mean(accs),
            "test_real_precision": torch.mean(precisions),
            "test_real_recal": torch.mean(recals),
            "test_real_f1": torch.mean(f1s),
        }

        grid = torch.cat(pred_images, 1)
        self.logger.experiment.add_image(
            "pred_real_data", grid, self.trainer.current_epoch
        )

        for key in tensorboard_logs:
            self.logger.experiment.add_scalar(
                key, tensorboard_logs[key], self.trainer.current_epoch
            )

        instance_model = self.to(torch.device("cuda"))
        instance_model.DGCNN_feature_space.device = "cuda"


class SorghumPartNetInstanceWithLeafBranch(pl.LightningModule):
    def __init__(self, hparams):
        """
        Parameters
        ----------
        hparams: hyper parameters
        """
        super(SorghumPartNetInstanceWithLeafBranch, self).__init__()

        self.hparams.update(hparams)
        self.lr_clip = 1e-5
        self.bnm_clip = 1e-2

        MyStruct = namedtuple("args", "k")
        args = MyStruct(k=15)

        self.DGCNN_feature_space = DGCNNFeatureSpace(args)

        leaf_classifier = SorghumPartNetLeaf.load_from_checkpoint(
            hparams["leaf_classifier_path"]
        )
        leaf_classifier = leaf_classifier.cuda()
        leaf_classifier.eval()

        self.leaf_branch = LeafBranch(hparams["leaf_space_threshold"], leaf_classifier)

        self.save_hyperparameters()

    def forward(self, xyz):

        # Normalization (min max)
        mins, _ = torch.min(xyz, axis=1)
        maxs, _ = torch.max(xyz, axis=1)
        mins = mins.unsqueeze(1)
        maxs = maxs.unsqueeze(1)
        xyz = (xyz - mins) / (maxs - mins) - 0.5

        # Instance
        dgcnn_features = self.DGCNN_feature_space(xyz)

        return dgcnn_features

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.lr_clip / self.hparams["lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["bn_momentum"]
            * self.hparams["bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self, ds_path, shuff=True):
        dataset = SorghumDataset(ds_path)
        loader = DataLoader(
            dataset, batch_size=self.hparams["batch_size"], num_workers=4, shuffle=shuff
        )
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["train_data"], shuff=True)

    def training_step(self, batch, batch_idx):
        points, _, _, _, leaf = batch

        pred_leaf_features = self(points)

        criterion_cluster = SpaceSimilarityLossV2()

        leaf_loss = criterion_cluster(pred_leaf_features, leaf)
        leaf_classifier_loss = self.leaf_branch(pred_leaf_features, points).cuda()
        loss = (
            leaf_loss + leaf_classifier_loss * self.hparams["leaf_classifier_loss_coef"]
        )

        leaf_metrics = LeafMetrics(self.hparams["leaf_space_threshold"])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features, leaf)

        tensorboard_logs = {
            "train_loss": loss.detach(),
            "train_leaf_loss": leaf_loss.detach(),
            "train_leaf_classifier_loss": leaf_classifier_loss.detach(),
            "train_leaf_accuracy": Acc,
            "train_leaf_precision": Prec,
            "train_leaf_recall": Rec,
            "train_leaf_f1": F,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {"loss": loss, "log": tensorboard_logs}

    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["val_data"], shuff=False)

    def validation_step(self, batch, batch_idx):
        points, _, _, _, leaf = batch

        pred_leaf_features = self(points)

        criterion_cluster = SpaceSimilarityLossV2()
        leaf_loss = criterion_cluster(pred_leaf_features, leaf)

        leaf_classifier_loss = self.leaf_branch(pred_leaf_features, points).cuda()
        loss = (
            leaf_loss + leaf_classifier_loss * self.hparams["leaf_classifier_loss_coef"]
        )

        leaf_metrics = LeafMetrics(self.hparams["leaf_space_threshold"])
        Acc, Prec, Rec, F = leaf_metrics(pred_leaf_features, leaf)

        tensorboard_logs = {
            "val_loss": loss.detach(),
            "val_leaf_loss": leaf_loss.detach(),
            "val_leaf_classifier_loss": leaf_classifier_loss.detach(),
            "val_leaf_accuracy": Acc,
            "val_leaf_precision": Prec,
            "val_leaf_recall": Rec,
            "val_leaf_f1": F,
        }

        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_leaf_loss = torch.stack([x["val_leaf_loss"] for x in outputs]).mean()
        val_leaf_classifier_loss = torch.stack(
            [x["val_leaf_classifier_loss"] for x in outputs]
        ).mean()
        Acc = torch.stack([x["val_leaf_accuracy"] for x in outputs]).mean()
        Prec = torch.stack([x["val_leaf_precision"] for x in outputs]).mean()
        Rec = torch.stack([x["val_leaf_recall"] for x in outputs]).mean()
        F = torch.stack([x["val_leaf_f1"] for x in outputs]).mean()

        tensorboard_logs = {
            "val_loss": val_loss.detach(),
            "val_leaf_loss": val_leaf_loss.detach(),
            "val_leaf_classifier_loss": val_leaf_classifier_loss.detach(),
            "val_leaf_accuracy": Acc,
            "val_leaf_precision": Prec,
            "val_leaf_recall": Rec,
            "val_leaf_f1": F,
        }

        for k in tensorboard_logs.keys():
            self.log(k, tensorboard_logs[k], on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": val_loss, "log": tensorboard_logs}


# The fine clustering is never merged. Their fine clustering is treated as
# the final / coarse clustering (look at the BCE loss between the final labels
# and the fine clustering predictions). Moreover, using integer labels and BCE
# inadvertantly enforces a meaning (distance/order) on the labels which is unwanted.
# The cluster numbers are solely for the sake of knowing which points are in the same
# Group.


class TreePartNet(pl.LightningModule):
    def __init__(self, hparams, debug=False):
        """
        Parameters
        ----------
        hparams: hyper parameters
        """
        super(TreePartNet, self).__init__()
        # self._hparams = hparams

        self.is_debug = debug
        self.hparams.update(hparams)
        self.lr_clip = 1e-5
        self.bnm_clip = 1e-2

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=hparams["lc_count"],
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[
                    [hparams["input_channels"], 16, 16, 16],
                    [hparams["input_channels"], 32, 32, 32],
                ],
                use_xyz=hparams["use_xyz"],
            )
        )

        c_out_0 = 16 + 32

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4],
                nsamples=[16, 32],
                mlps=[[c_out_0, 64, 64, 128], [c_out_0, 64, 64, 128]],
                use_xyz=hparams["use_xyz"],
            )
        )

        c_out_1 = 128 + 128

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[64, 256, 64]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_0 + c_out_1, 256, 64]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 2, kernel_size=1),
        )

        self.lp_fc_layer = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 3, kernel_size=1),
        )

        self.sharedMLP_layer = build_shared_mlp([64, 64, 32, 1])
        self.dot = ScaledDot(64)
        self.scale = nn.Parameter(
            torch.tensor(10.0, dtype=torch.float), requires_grad=True
        )

        self.save_hyperparameters()

    def forward(self, xyz):
        num_point = xyz.shape[1]

        # PointNet SA Module
        l_xyz, l_features, l_s_idx = [xyz], [None], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_s_idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_s_idx.append(li_s_idx)

        # PointNet FP Module
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # Local Context Label Prediction
        point_feat = torch.unsqueeze(l_features[0], dim=-2)
        point_feat = point_feat.repeat(1, 1, self.hparams["lc_count"], 1)
        lc_feat = torch.unsqueeze(l_features[-2], dim=-1)
        lc_feat = lc_feat.repeat(1, 1, 1, num_point)
        per_point_feat = point_feat - lc_feat

        lc_pred = self.sharedMLP_layer(per_point_feat)
        lc_pred = lc_pred.squeeze(dim=1)

        # Tree Edge Prediction
        dot = self.dot(l_features[-2])  # Cosine similarity
        batch_idx = torch.tensor(range(xyz.shape[0]))
        batch_idx = batch_idx.unsqueeze(-1)
        batch_idx = batch_idx.repeat(1, self.hparams["lc_count"])
        s_xyz = xyz[batch_idx.cuda().long(), l_s_idx[-2].long()]
        s_xyz = torch.unsqueeze(s_xyz, dim=-2)
        s_xyz = s_xyz.repeat(1, 1, self.hparams["lc_count"], 1)

        dis = s_xyz - s_xyz.permute(0, 2, 1, 3)
        dis = dis**2
        dis = torch.sum(dis, dim=-1)
        dis = torch.sqrt(dis)  # Euclidean distance

        fnode_pred = dot - self.scale * dis

        # Sampled Points for cluster merging
        lc_idx = l_s_idx[0]
        lc_idx = lc_idx[:, 0 : self.hparams["lc_count"]]

        return lc_pred, fnode_pred, lc_idx

    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.hparams["lr_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.lr_clip / self.hparams["lr"],
        )
        bn_lbmd = lambda _: max(
            self.hparams["bn_momentum"]
            * self.hparams["bnm_decay"]
            ** (
                int(
                    self.global_step
                    * self.hparams["batch_size"]
                    / self.hparams["decay_step"]
                )
            ),
            self.bnm_clip,
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)
        bnm_scheduler.optimizer = optimizer

        return [optimizer], [lr_scheduler, bnm_scheduler]

    def _build_dataloader(self, ds_path, shuff=True):
        if self.hparams["dataset"] == "SPN":
            dataset = SorghumDatasetTPNFormat(
                ds_path,
                self.hparams["std_noise"],
                debug=self.is_debug,
            )
        elif self.hparams["dataset"] == "TPN":
            dataset = TreePartNetOriginalDataset(
                ds_path,
                debug=self.is_debug,
            )
        elif self.hparams["dataset"] == "PN":
            # not working
            dataset = PartNetDataset(
                ds_path,
                debug=self.is_debug,
            )

        loader = DataLoader(
            dataset, batch_size=self.hparams["batch_size"], num_workers=4, shuffle=shuff
        )
        return loader

    def train_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["train_data"], shuff=True)

    def training_step(self, batch, batch_idx):
        pxyz, lcl, fn = batch
        pred_lcl, fnode_pred, _ = self(pxyz)
        critirion = torch.nn.CrossEntropyLoss()
        lc_loss = critirion(pred_lcl, lcl)
        critirion2 = FocalLoss(
            alpha=self.hparams["FL_alpha"],
            gamma=self.hparams["FL_gamma"],
            reduce="mean",
        )
        fn_loss = critirion2(fnode_pred, fn)
        total_loss = lc_loss + fn_loss

        with torch.no_grad():

            leaf_metrics = LeafMetrics("cuda")
            acc, prec, recal, f1 = leaf_metrics(
                torch.argmax(pred_lcl, dim=1).unsqueeze(-1), lcl.unsqueeze(-1)
            )
            tensorboard_logs = {
                "train_loss": total_loss,
                "train_local_context_loss": lc_loss,
                "train_fnode_loss": fn_loss,
                "train_local_context_acc": acc,
                "train_local_context_prec": prec,
                "train_local_context_recal": recal,
                "train_local_context_f1": f1,
            }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return {"loss": total_loss, "log": tensorboard_logs}

    def val_dataloader(self):
        return self._build_dataloader(ds_path=self.hparams["val_data"], shuff=False)

    def validation_step(self, batch, batch_idx):
        pxyz, lcl, fn = batch
        pred_lcl, fnode_pred, _ = self(pxyz)
        critirion = torch.nn.CrossEntropyLoss()
        lc_loss = critirion(pred_lcl, lcl)
        critirion2 = FocalLoss(
            alpha=self.hparams["FL_alpha"],
            gamma=self.hparams["FL_gamma"],
            reduce="mean",
        )
        fn_loss = critirion2(fnode_pred, fn)
        total_loss = lc_loss + fn_loss

        leaf_metrics = LeafMetrics("cuda")
        acc, prec, recal, f1 = leaf_metrics(
            torch.argmax(pred_lcl, dim=1).unsqueeze(-1), lcl.unsqueeze(-1)
        )
        tensorboard_logs = {
            "val_loss": total_loss,
            "val_local_context_loss": lc_loss,
            "val_fnode_loss": fn_loss,
            "val_local_context_acc": acc,
            "val_local_context_prec": prec,
            "val_local_context_recal": recal,
            "val_local_context_f1": f1,
        }

        for k in tensorboard_logs.keys():
            self.log(
                k,
                tensorboard_logs[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return tensorboard_logs
