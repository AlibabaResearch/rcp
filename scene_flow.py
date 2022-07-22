from typing import List, Union
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from losses.common_losses import curvature, curvatureWarp

from utils.metrics import SceneFlowMetrics
from utils.utils import get_num_workers
from data.exp_utils import get_datasets
from losses import *
from lib.pointnet2 import pointnet2_utils as pointutils


class SceneFlowModel(pl.LightningModule):

    def __init__(self, model, cfg):
        """
        Initializes an experiment.

        Arguments:
            model : a torch.nn.Module object, model to be tested
            hparams : a dictionary of hyper parameters
        """

        super(SceneFlowModel, self).__init__()
        self.model = model
        self.save_hyperparameters()
        
        self.loss = losses_dict[cfg["exp_params"]['loss']['loss_type']](**cfg["exp_params"]['loss'])
        
        self.train_metrics = SceneFlowMetrics(split='train', loss_params=cfg["exp_params"]['loss'], reduce_op='mean')
        self.val_metrics = SceneFlowMetrics(split='val', loss_params=cfg["exp_params"]['loss'], reduce_op='mean')

    def forward(self, pos1, pos2, feat1, feat2, iters):
        """
        A forward call
        """
        return self.model(pos1, pos2, feat1, feat2, iters)

    def sequence_loss(self, pos1, pos2, flows_pred, flow_gt):
        if 'loss_iters_w' in self.hparams["cfg"]["exp_params"]:
            assert (len(self.hparams["cfg"]["exp_params"]['loss_iters_w']) == len(flows_pred)) or (len(self.hparams["cfg"]["exp_params"]['loss_iters_w']) == 2)
            if len(self.hparams["cfg"]["exp_params"]['loss_iters_w']) != 2:
                w_new = self.hparams["cfg"]["exp_params"]['loss_iters_w']
            else:
                gamma = self.hparams["cfg"]["exp_params"]['loss_gamma']
                w_new = get_loss_weights(self.hparams["cfg"]["exp_params"]['loss_iters_w'], seq_len=len(flows_pred), gamma=gamma)

            loss = torch.zeros(1).cuda()
            for i, w in enumerate(w_new):
                loss += w * self.loss(pos1, pos2, flows_pred[i], flow_gt, len(flows_pred), i)
        else:
            loss = self.loss(pos1, pos2, flows_pred[-1], flow_gt)
        return loss
        
    def training_step(self, batch, batch_idx):
        """
        Executes a single training step
        """
        pos1, pos2, feat1, feat2, flow_gt, fnames = batch
        flows_pred = self(pos1, pos2, feat1, feat2, self.hparams["cfg"]["exp_params"]['train_iters'])
        loss = self.sequence_loss(pos1, pos2, flows_pred, flow_gt)
        metrics = self.train_metrics(pos1, pos2, flows_pred, flow_gt)

        i_last = self.hparams["cfg"]["exp_params"]['train_iters'] - 1
        train_epe = metrics[f'train_epe3d_i#{i_last}']

        # self.log('train_loss', loss, sync_dist=False, prog_bar=False, on_step=False, on_epoch=True, logger=True)
        self.log('train_epe', train_epe, sync_dist=False, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log_dict(metrics, sync_dist=False, prog_bar=False, on_step=False, on_epoch=True, logger=True)  # No need to sync_dist since metrics are already synced 

        return {"loss": loss, "metrics": metrics}


    def _test_val_step(self, batch, batch_idx, split):
        pos1, pos2, feat1, feat2, flow_gt, fnames = batch #(B, N, 3)
        flows_pred = self(pos1, pos2, feat1, feat2, self.hparams["cfg"]["exp_params"][f'{split}_iters'])
        loss = self.sequence_loss(pos1, pos2, flows_pred, flow_gt)
        metrics = self.val_metrics(pos1, pos2, flows_pred, flow_gt)
        
        i_last = self.hparams["cfg"]["exp_params"][f'{split}_iters'] - 1
        val_epe = metrics[f'val_epe3d_i#{i_last}']
        metrics_last = {k: v for k, v in metrics.items() if f"i#{i_last}" in k}

        self.log('val_epe', val_epe, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, logger=True)
        # self.log_dict(metrics, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True, logger=True)  # No need to sync_dist since metrics are already synced
        self.log_dict(metrics_last, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True, logger=True)  # No need to sync_dist since metrics are already synced

        return {"val_epe": metrics[f'val_epe3d_i#{i_last}'], "val_loss": loss}

    def _test_val_epoch_end(self, outputs):
        val_epe = torch.stack([e['val_epe'] for e in outputs], dim=0).mean(0)
        # val_loss = torch.stack([e['val_loss'] for e in outputs], dim=0).mean(0)
        if len(self.trainer.lr_schedulers) > 0:
            print(f"**********val_epe: {val_epe}, lr:{self.trainer.lr_schedulers[0]['scheduler'].get_last_lr()}")
        else:
            print(f"**********val_epe: {val_epe}")
        self.log("val_epe_epoch", value=val_epe, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, logger=True)
        # self.log("val_loss_epoch", value=val_loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, logger=True)
        return None

    def test_epoch_end(self, outputs):
        return self._test_val_epoch_end(outputs)
    
    def validation_epoch_end(self, outputs):
        return self._test_val_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        """
        Executes a single validation step.
        """
        return self._test_val_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        """
        Executes a single test step.
        """
        return self._test_val_step(batch, batch_idx, 'test')

    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams["cfg"]["exp_params"]["optimizer"]["type"])
        # Create optimizer with parameters
        params = self.hparams["cfg"]["exp_params"]["optimizer"]
        params.pop("type", None)
        optimizer = optimizer(self.parameters(), **params)
        
        # Load and initialize scheduler
        scheduler = getattr(torch.optim.lr_scheduler, self.hparams["cfg"]["exp_params"]["scheduler"]["type"])
        params = self.hparams["cfg"]["exp_params"]["scheduler"]
        params.pop("type", None)
        scheduler = scheduler(optimizer,  **params)

        return {'optimizer': optimizer,
                'lr_scheduler': scheduler}

    def train_dataloader(self):
        """
        Returns a train set Dataloader object 
        """
        return self._dataloader(split_type='train')
        
    def val_dataloader(self):
        """
        Returns a validation set Dataloader object 
        """
        return self._dataloader(split_type='val')

    def test_dataloader(self):
        """
        Returns a validation set Dataloader object
        """
        return self._dataloader(split_type='test')


    def setup(self, stage: str):
        """
        Load datasets
        """
        train_dataset, val_dataset, test_dataset = get_datasets(task="sceneflow", data_params=self.hparams["cfg"]["exp_params"]['data'])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def _dataloader(self, split_type: str):
        """
        Arguments:
            split_type : (str) shouild be one of ['train', 'val', 'test']
        Return:
            A DataLoader object of the correspondig split
        """
        
        split_dict = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
            }
        is_train = (split_type == 'train')
        num_workers = get_num_workers(self.hparams["cfg"]["exp_params"]['num_workers'])
        if split_dict[split_type] is None:
            loader = None
        else:
            loader = DataLoader(split_dict[split_type],
                                batch_size=self.hparams["cfg"]["exp_params"]['batch_size'],
                                num_workers=num_workers,
                                pin_memory=True,
                                shuffle=True if is_train else False)
        return loader

