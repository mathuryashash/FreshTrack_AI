
import torch
import torch.nn as nn
import timm
import pytorch_lightning as pl

class FreshTrackModel(pl.LightningModule):
    def __init__(self, num_freshness=4, num_quality=3, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Backbone
        # Using EfficientNet-B0 as per PRD
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,  # Remove head
            global_pool='avg'
        )
        
        in_features = 1280 # EfficientNet-B0 feature size
        
        # Multi-task heads
        self.freshness_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_freshness)
        )
        
        self.quality_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_quality)
        )
        
        self.shelf_life_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()  # Positive days only
        )
        
        # Auxiliary task (rotation prediction) mentioned in workflow
        self.rotation_head = nn.Linear(in_features, 4)
        
        # Loss weights
        self.w_fresh = 0.4
        self.w_quality = 0.3
        self.w_shelf = 0.25
        self.w_rot = 0.05
    
    def forward(self, x):
        features = self.backbone(x)
        
        freshness_logits = self.freshness_head(features)
        quality_logits = self.quality_head(features)
        shelf_life = self.shelf_life_head(features)
        rotation_logits = self.rotation_head(features)
        
        return freshness_logits, quality_logits, shelf_life, rotation_logits
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        fresh_logits, qual_logits, shelf_pred, rot_logits = self(images)
        
        # Losses
        loss_fresh = nn.CrossEntropyLoss()(fresh_logits, labels['freshness'])
        loss_qual = nn.CrossEntropyLoss()(qual_logits, labels['quality'])
        # Ensure shelf_life shapes match
        loss_shelf = nn.MSELoss()(shelf_pred, labels['shelf_life'])
        loss_rot = nn.CrossEntropyLoss()(rot_logits, labels['rotation'])
        
        # Combined loss
        total_loss = (
            self.w_fresh * loss_fresh +
            self.w_quality * loss_qual +
            self.w_shelf * loss_shelf +
            self.w_rot * loss_rot
        )
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_loss_fresh', loss_fresh)
        self.log('train_loss_quality', loss_qual)
        self.log('train_loss_shelf', loss_shelf)
        
        # Accuracy
        fresh_acc = (fresh_logits.argmax(dim=1) == labels['freshness']).float().mean()
        self.log('train_acc_fresh', fresh_acc, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        fresh_logits, qual_logits, shelf_pred, rot_logits = self(images)
        
        # Losses
        loss_fresh = nn.CrossEntropyLoss()(fresh_logits, labels['freshness'])
        loss_qual = nn.CrossEntropyLoss()(qual_logits, labels['quality'])
        loss_shelf = nn.MSELoss()(shelf_pred, labels['shelf_life'])
        
        total_loss = (
            self.w_fresh * loss_fresh +
            self.w_quality * loss_qual +
            self.w_shelf * loss_shelf
        )
        
        # Metrics
        fresh_acc = (fresh_logits.argmax(dim=1) == labels['freshness']).float().mean()
        
        # MAE for shelf life
        mae_shelf = torch.abs(shelf_pred - labels['shelf_life']).mean()
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_acc_fresh', fresh_acc, prog_bar=True)
        self.log('val_mae_shelf', mae_shelf, prog_bar=True)
        
        return total_loss
    
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        
        fresh_logits, qual_logits, shelf_pred, rot_logits = self(images)
        
        # Losses
        loss_fresh = nn.CrossEntropyLoss()(fresh_logits, labels['freshness'])
        loss_qual = nn.CrossEntropyLoss()(qual_logits, labels['quality'])
        loss_shelf = nn.MSELoss()(shelf_pred, labels['shelf_life'])
        
        total_loss = (
            self.w_fresh * loss_fresh +
            self.w_quality * loss_qual +
            self.w_shelf * loss_shelf
        )
        
        # Metrics
        fresh_acc = (fresh_logits.argmax(dim=1) == labels['freshness']).float().mean()
        
        # MAE for shelf life
        mae_shelf = torch.abs(shelf_pred - labels['shelf_life']).mean()
        
        self.log('test_loss', total_loss, prog_bar=True)
        self.log('test_acc_fresh', fresh_acc, prog_bar=True)
        self.log('test_mae_shelf', mae_shelf, prog_bar=True)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
