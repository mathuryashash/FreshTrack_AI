import torch
import torch.nn as nn
import torchvision.models as models

class multi_task_fruit_model(nn.Module):
    """
    Multi-task CNN for Fruit Freshness, Quality Grading, and Shelf-life Prediction.
    Architecture: Backbone (ResNet18) -> Shared Features -> Task-specific Heads
    """
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super(multi_task_fruit_model, self).__init__()
        
        # 1. Backbone selection
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            # Remove the original FC layer
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained).features
            self.backbone.classifier = nn.Identity()
            self.feature_dim = 1280
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Backbone {backbone_name} not supported.")
        
        self.backbone_name = backbone_name

        # 2. Freshness Head (Classification: Fresh, Semi-ripe, Overripe, Rotten)
        self.freshness_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 4)
        )
        
        # 3. Quality Head (Classification: A, B, C)
        self.quality_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
        
        # 4. Shelf-life Head (Regression)
        # Includes meta-data (temp, humidity) injection
        self.meta_processor = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        
        self.shelf_life_head = nn.Sequential(
            nn.Linear(self.feature_dim + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, image, meta=None):
        """
        Forward pass for the multi-task model.
        Args:
            image (torch.Tensor): Input image tensor of shape (B, 3, H, W)
            meta (torch.Tensor, optional): Metadata tensor (temp, humidity) of shape (B, 2)
        Returns:
            dict: Predictions for freshness, quality, and shelf_life
        """
        # Feature extraction
        if self.backbone_name == 'resnet18':
            features = self.backbone(image)
        else:
            features = self.backbone(image)
            features = self.pool(features)
            features = torch.flatten(features, 1)
            
        # Freshness prediction
        freshness_logits = self.freshness_head(features)
        
        # Quality prediction
        quality_logits = self.quality_head(features)
        
        # Shelf-life prediction (Regression)
        if meta is None:
            # Default meta values if not provided (placeholder values)
            meta = torch.zeros((image.size(0), 2)).to(image.device)
            
        meta_features = self.meta_processor(meta)
        combined_features = torch.cat((features, meta_features), dim=1)
        shelf_life = self.shelf_life_head(combined_features)
        
        return {
            'freshness': freshness_logits,
            'quality': quality_logits,
            'shelf_life': shelf_life
        }

if __name__ == "__main__":
    # Test the model with dummy input
    model = multi_task_fruit_model(backbone_name='resnet18', pretrained=False)
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_meta = torch.randn(2, 2) # [temp, humidity]
    
    outputs = model(dummy_image, dummy_meta)
    
    print(f"Freshness output shape: {outputs['freshness'].shape}") # Expect [2, 4]
    print(f"Quality output shape: {outputs['quality'].shape}")     # Expect [2, 3]
    print(f"Shelf-life output shape: {outputs['shelf_life'].shape}") # Expect [2, 1]
