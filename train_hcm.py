

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
from tqdm import tqdm
import os
import math
import random
import glob
import pandas as pd

from preprocess import z_normalize, resample_cmr, pad_3d, fast_resample_data_or_seg_to_shape

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from models.model import Model, CustomLoss
from models.model_cfg import CFG
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim import AdamW
import SimpleITK as sitk
import transformers
import pytorch_lightning as pl





def split_data(files):
    df = pd.read_csv('/home/jma/Documents/yu/HCM/GenomeVariables.csv')
    #print(df.columns)
    data_splits = {}
    for file in files:
        patient_id = int(os.path.basename(file).split('_')[1])
        # idx = df.loc[df['Patient ID'] == patient_id, 'Septal reduction therapy '].values[0]
        idx = df.loc[df['Patient ID'] == patient_id, 'A-Fib'].values[0]
        idx = int(os.path.basename(file).split('_')[0])
        if idx in data_splits:
            data_splits[idx].append(file)
        else:
            data_splits[idx] = [file]
    train_split = []
    valid_split = []
    for key, value in data_splits.items():
        print(key, len(value))
        random.shuffle(value)
        split_index = int(len(value) * 0.8)
        train_split.extend(value[:split_index])
        valid_split.extend(value[split_index:])
    return train_split, valid_split




class HCMDataset(Dataset):
    def __init__(self, data, seg_root, transforms, is_training):
        self.data = data
        self.seg_root = seg_root
        self.transforms = transforms
        self.is_training = is_training
        self.df = pd.read_csv('/home/jma/Documents/yu/HCM/GenomeVariables.csv')


    def preprocess(self, image, segmentation):
        image = resample_cmr(image)
        segmentation = resample_cmr(segmentation, is_label=True)
        image = z_normalize(image)
        image = pad_3d(image)
        segmentation = pad_3d(segmentation)
        image = fast_resample_data_or_seg_to_shape(image.unsqueeze(0), (25,128,128))[0]
        segmentation = fast_resample_data_or_seg_to_shape(segmentation.unsqueeze(0), (25,128,128), True)[0]
        image = image.numpy()
        segmentation = segmentation.numpy()
        image = np.repeat(image[:, np.newaxis, :, :], 3, axis=1)  # Repeat along the channel dimension
        segmentation = np.repeat(segmentation[:, np.newaxis, :, :], 3, axis=1)

        return image, segmentation


    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        
        image_path = self.data[i]
        seg_path = os.path.join(self.seg_root, os.path.basename(image_path).replace('_0000.nii.gz', '.nii.gz'))
        image = sitk.ReadImage(image_path)
        segmentation_volume = sitk.ReadImage(seg_path)

        processed_image, processed_seg = self.preprocess(image, segmentation_volume)
        volume_, mask_volume = [], []
        if self.transforms:
            first = True
            for image, mask in zip(processed_image, processed_seg):
                
                
                #image = image.astype(np.float32) / 255
                if self.is_training:
                    if first:
                        transformed = self.transforms(image=image, mask=mask)
                        replay = transformed['replay']
                        first = False
                    else:
                        transformed = A.ReplayCompose.replay(replay, image=image, mask=mask)
                else:
                    transformed = self.transforms(image=image, mask=mask)
                
                image = transformed['image']
                mask = transformed['mask']
                
                volume_.append(image)
                mask_volume.append(mask)
            volume = np.stack(volume_)
            mask_volume = np.stack(mask_volume).transpose(0, 3, 1, 2)
            volume = volume.astype(np.float32)
        else:
            volume = torch.from_numpy(processed_image)
            mask_volume = torch.from_numpy(processed_seg)

        
        #print(volume.shape, mask_volume.shape)
        
        patient_id = int(os.path.basename(image_path).split('_')[1])
        label = int(self.df.loc[self.df['Patient ID'] == patient_id, 'A-Fib'].values[0])
        #label = int(self.df.loc[self.df['Patient ID'] == patient_id, 'Septal reduction therapy '].values[0])
        #label = int(os.path.basename(image_path).split('_')[0])
        num_classes = 2  # Adjust this based on the number of classes in your dataset
        one_hot_label = np.zeros(num_classes, dtype=np.float32)
        one_hot_label[label] = 1.0

        
        return {'images': volume,
                'labels': one_hot_label,
                'masks': mask_volume}


def get_loaders(path, seg_root):
    
    files = glob.glob(os.path.join(path, '*.nii.gz'))
    train_volumes, valid_volumes = split_data(files)
    import json

    # Save valid_volumes as a JSON file
    valid_volumes_json_path = 'valid_volumes.json'
    with open(valid_volumes_json_path, 'w') as json_file:
        json.dump(valid_volumes, json_file)
    
    train_dataset = HCMDataset(train_volumes, seg_root, None, 1)
    valid_dataset = HCMDataset(valid_volumes, seg_root, None, 0)
    
    print(len(train_dataset), len(valid_dataset))
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=CFG.workers, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.workers, pin_memory=False)
    
    CFG.steps_per_epoch = math.ceil(len(train_loader) / CFG.acc_steps)
    
    return train_loader, valid_loader

def get_valid_loader(seg_root):
    import json
    with open('/home/jma/Documents/yu/cnn_lstm/valid_volumes.json', 'r') as jsonfile:
        valid_volumes = json.load(jsonfile)
    valid_dataset = HCMDataset(valid_volumes, seg_root, None, 0)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.workers, pin_memory=False)
    return valid_loader



VAL_SCORE = []
class ClassModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = CustomLoss()
        self.model = Model(class_num=2)

        self.val_step_outputs = []
        self.val_step_labels = []

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
        images, masks, labels = batch['images'], batch['masks'], batch['labels']
        #print(images.shape)
        preds, segmentations = self.model(images)
        #print(preds.shape, labels.shape)
        loss = self.loss(preds, labels, segmentations, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.config['train_bs'])

        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Add TTA"""

        images, masks, labels = batch['images'], batch['masks'], batch['labels']
        #print('valid:', images.shape)
        preds, segmentations = self.model(images)
        loss = self.loss(preds, labels, segmentations, masks)
        self.log("val_loss_mean", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(preds)
        self.val_step_labels.append(labels)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_outputs).float()
        all_labels = torch.cat(self.val_step_labels)
        all_labels = all_labels.to(torch.long)

        # Calculate accuracy
        preds_labels = torch.argmax(all_preds, dim=1)
        all_labels = torch.argmax(all_labels, dim=1)
        #print(preds_labels, all_labels)
        accuracy = (preds_labels == all_labels).float().mean()
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        print("Validation Accuracy:", accuracy.item())
        
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
    with open("config.yaml", "r") as file_obj:
        config = yaml.safe_load(file_obj)
    image_folder = '/home/jma/Documents/yu/HCM/sax_cine_train/roi'
    seg_folder = '/home/jma/Documents/yu/HCM/sax_cine_train/seg'
    train_loader, valid_loader = get_loaders(image_folder, seg_folder)
    #valid_loader = get_valid_loader(seg_folder)
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="val_accuracy",
        dirpath=config["output_dir"],
        mode='max',
        filename=f"Afib_best",
        save_top_k=config["save_topk"],
        verbose=1,
    )

    progress_bar_callback = TQDMProgressBar(
        refresh_rate=config["progress_bar_refresh_rate"]
    )


    wandb_logger = WandbLogger(project=f'cnn_lstm', # group runs in "MNIST" project
                            log_model=False) # log all new checkpoints during training

    trainer = pl.Trainer(
        val_check_interval=108,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, progress_bar_callback],
        **config["trainer"],
    )
    num_training_steps = config["model"]["scheduler"]["params"]["cosine_with_warmup"]["num_training_steps"]
    num_warmup_steps = config["model"]["scheduler"]["params"]["cosine_with_warmup"]["num_warmup_steps"]
    config["model"]["scheduler"]["params"]["cosine_with_warmup"]["num_training_steps"] = int(num_training_steps*len(train_loader)/config["trainer"]["devices"])
    config["model"]["scheduler"]["params"]["cosine_with_warmup"]["num_warmup_steps"] = int(num_warmup_steps*len(train_loader)/config["trainer"]["devices"])
    model = ClassModule(config=config)
    trainer.fit(model, train_loader, valid_loader)
    wandb.finish()