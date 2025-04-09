

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
  

    
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import argparse
from tqdm import tqdm
import os
import math
import random
import glob

from datasets.mp_liver_dataset import MultiPhaseLiverDataset, create_loader
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from models.model import Model, CustomLoss
from models.model_cfg import CFG
from torch.optim import AdamW
import transformers
import pytorch_lightning as pl





VAL_SCORE = []
class ClassModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = CustomLoss(mask_head=False)
        self.model = Model(in_chans=8, class_num=7, mask_head=False)

        self.val_step_outputs = []
        self.val_step_labels = []
        self.val_step_ids = []

    def forward(self, batch):
        preds, masks = self.model(batch)
        return preds, masks

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.config['model']["optimizer_params"])
        
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            **self.config['model']['scheduler']['params']['cosine_with_warmup'],
        )
        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

    def training_step(self, batch, batch_idx):
        (images, labels, patient_id) = batch
        self.patient_id = patient_id
        preds, segmentations = self.model(images)
        #print(preds.shape, labels.shape)
        loss = self.loss(preds, labels, segmentations, None)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.config['train_bs'])

        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Add TTA"""

        (images, labels, patient_id) = batch
        self.patient_id = patient_id
        preds, segmentations = self.model(images)
        loss = self.loss(preds, labels, segmentations, None)
        self.log("val_loss_mean", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(preds)
        self.val_step_labels.append(labels)
        self.val_step_ids.append(patient_id)

    def on_validation_epoch_end(self):

        all_preds = torch.cat(self.val_step_outputs).float()
        all_labels = torch.cat(self.val_step_labels)
        all_labels = all_labels.to(torch.long)

        # Calculate accuracy
        preds_labels = torch.argmax(all_preds, dim=1)
        all_labels = torch.argmax(all_labels, dim=1)
        #print(preds_labels.shape, all_labels.shape, all_ids.shape)
        accuracy = (preds_labels == all_labels).float().mean()
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        print("Validation Accuracy:", accuracy.item())

        # print("Validation outputs and labels saved as a DataFrame.")
        self.val_step_outputs.clear()
        self.val_step_labels.clear()

        print(f"\nEpoch: {self.current_epoch}", flush=True)
        return

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)





if __name__ == '__main__':
    import yaml
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        '--data_dir', default='/home/jma/Documents/yu/LLD_MMRI/imgs_pre255', type=str)
    parser.add_argument(
        '--train_anno_file', default='/home/jma/Documents/yu/LLD_MMRI/LLD_MMRI_splits/train.txt', type=str)
    parser.add_argument(
        '--val_anno_file', default='/home/jma/Documents/yu/LLD_MMRI/LLD_MMRI_splits/test.txt', type=str)
    parser.add_argument('--train_transform_list', default=['random_crop',
                                                           'z_flip',
                                                           'x_flip',
                                                           'y_flip',
                                                           'rotation',
                                                           'edge',
                                                            'emboss',
                                                            'filter' ],
                        nargs='+', type=str)
    parser.add_argument('--val_transform_list',
                        default=['center_crop'], nargs='+', type=str)
    parser.add_argument('--img_size', default=(16, 128, 128),
                        type=int, nargs='+', help='input image size.')
    parser.add_argument('--crop_size', default=(14, 112, 112),
                        type=int, nargs='+', help='cropped image size.')
    parser.add_argument('--flip_prob', default=0.5, type=float,
                        help='Random flip prob (default: 0.5)')
    parser.add_argument('--angle', default=45, type=int)
    parser.add_argument('--mode', default='trilinear')
    parser.add_argument('--mixup', default=True)
    args = parser.parse_args()

    with open("config.yaml", "r") as file_obj:
        config = yaml.safe_load(file_obj)
    train_dataset = MultiPhaseLiverDataset(args, is_training=True)
    train_loader = create_loader(train_dataset, batch_size=4, is_training=True)
    valid_dataset = MultiPhaseLiverDataset(args, is_training=False)
    valid_loader = create_loader(valid_dataset, batch_size=4, is_training=False)
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="val_accuracy",
        dirpath=config["output_dir"],
        mode='max',
        filename=f"LLD_best",
        save_top_k=config["save_topk"],
        verbose=1,
    )

    progress_bar_callback = TQDMProgressBar(
        refresh_rate=config["progress_bar_refresh_rate"]
    )


    wandb_logger = WandbLogger(project=f'cnn_lstm', # group runs in "MNIST" project
                            log_model=False) # log all new checkpoints during training

    trainer = pl.Trainer(
        val_check_interval=1.0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, progress_bar_callback],
        **config["trainer"],
    )
    num_training_steps = config["model"]["scheduler"]["params"]["cosine_with_warmup"]["num_training_steps"]
    num_warmup_steps = config["model"]["scheduler"]["params"]["cosine_with_warmup"]["num_warmup_steps"]
    config["model"]["scheduler"]["params"]["cosine_with_warmup"]["num_training_steps"] = int(num_training_steps*len(train_loader)/config["trainer"]["devices"])
    config["model"]["scheduler"]["params"]["cosine_with_warmup"]["num_warmup_steps"] = int(num_warmup_steps*len(train_loader)/config["trainer"]["devices"])
    model = ClassModule(config=config)
    #trainer.fit(model, train_loader, valid_loader)
    trainer.validate(model=model, dataloaders=valid_loader, ckpt_path='/home/jma/Documents/yu/cnn_lstm/checkpoints/LLD_best-v3.ckpt', )
    wandb.finish()