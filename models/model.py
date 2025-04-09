import torch
from torch import nn, optim
import torch.nn.functional as F

import timm
import segmentation_models_pytorch as smp

from models.model_cfg import CFG

from transformers import get_cosine_schedule_with_warmup

class LSTMMIL(nn.Module):
    def __init__(self, input_dim):
        super(LSTMMIL, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim//2, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.aux_attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        self.attention = nn.Sequential(
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    def forward(self, bags):
        """
        Args:
            bags: (batch_size, num_instances, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size, num_instances, input_dim = bags.size()
        bags_lstm, _ = self.lstm(bags)
        attn_scores = self.attention(bags_lstm).squeeze(-1)
        aux_attn_scores = self.aux_attention(bags_lstm).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_instances = torch.bmm(attn_weights.unsqueeze(1), bags_lstm).squeeze(1)  # (batch_size, input_dim)
        return weighted_instances, aux_attn_scores

class Model(nn.Module):
    def __init__(self, pretrained=True, in_chans=3, class_num=3, mask_head=True, seg_class_num=3):
        super(Model, self).__init__()
        
        self.mask_head = mask_head
        
        drop = 0.
        
        true_encoder = timm.create_model(CFG.model_name, pretrained=pretrained, in_chans=in_chans, global_pool='', num_classes=0, drop_rate=drop, drop_path_rate=drop)

        segmentor = smp.Unet(f"tu-{CFG.model_name}", encoder_weights='imagenet', in_channels=in_chans, classes=seg_class_num)
        self.encoder = segmentor.encoder
        self.decoder = segmentor.decoder
        self.segmentation_head = segmentor.segmentation_head
        
        st = true_encoder.state_dict()

        self.encoder.model.load_state_dict(st, strict=False)
        
        self.conv_head = true_encoder.conv_head
        self.bn2 = true_encoder.bn2
        
        feats = true_encoder.num_features
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        lstm_embed = feats * 1
        
        self.lstm = LSTMMIL(lstm_embed)#nn.LSTM(lstm_embed, lstm_embed//2, num_layers=1, dropout=drop, bidirectional=True, batch_first=True)
        
        self.head = nn.Sequential(
            #nn.Linear(lstm_embed, lstm_embed//2),
            #nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(0.1),
            #nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed, class_num),
        )

    def forward(self, x):  # (bs, nslice, ch, sz, sz)
        
        x = torch.nan_to_num(x, 0, 0, 0)
        
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        
        features = self.encoder(x)
        
        if self.mask_head:
        
            decoded = self.decoder(*features)
        
            masks = self.segmentation_head(decoded)
        
        feat = features[-1]
        feat = self.conv_head(feat)
        feat = self.bn2(feat)
        
        avg_feat = self.avgpool(feat)
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        
        feat = avg_feat
        
        #max_feat = self.maxpool(feat)
        #max_feat = max_feat.view(bs, n_slice_per_c, -1)
        
        #feat = torch.cat([avg_feat, max_feat], -1)
        
        feat, _ = self.lstm(feat)
        #print(feat.shape)
        #feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        
        feat = self.head(feat)
        
        #feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        #print(feat.shape)
        feat = torch.nan_to_num(feat, 0, 0, 0)
        
        if self.mask_head:
            return feat, masks
        else:
            return feat, None
        

class CustomLoss(nn.Module):
    def __init__(self, mask_head=True):
        super(CustomLoss, self).__init__()
        #self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([2.318]).cuda()).cuda()
        self.bce = nn.BCEWithLogitsLoss()
        #self.bce =  nn.BCEWithLogitsLoss(weight=torch.as_tensor([1, 1, 1, 2, 4, 2, 4, 2, 4]).cuda())
        self.dice = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)
        self.mask_head = mask_head
        
    def forward(self, outputs, targets, masks_outputs, masks_targets):
        
        loss1 = self.bce(outputs, targets)
        
        if self.mask_head:
            masks_outputs = masks_outputs.float()
            masks_targets = masks_targets.float().flatten(0, 1)
            loss2 = self.dice(masks_outputs, masks_targets)
            
            loss = loss1 + (loss2 * 0.1)
        else:
            loss = loss1
        
        return loss

def define_criterion_optimizer_scheduler_scaler(model, CFG):
    criterion = CustomLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=CFG.steps_per_epoch * CFG.n_epochs * CFG.upscale_steps, num_warmup_steps=CFG.n_warmup_steps)
    
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.autocast)
    
    return criterion, optimizer, scheduler, scaler


if __name__ == "__main__":
    model = Model()  # Replace YourModelClass with the actual model class name
    random_input = torch.randn(2, 32, 3, 224, 224)  # Create a random input tensor
    output = model(random_input)  # Forward pass through the model
    print("Output shape:", output[0].shape if isinstance(output, tuple) else output.shape)  # Print the output shape

