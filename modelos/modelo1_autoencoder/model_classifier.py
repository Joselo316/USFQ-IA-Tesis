"""
Clasificador binario basado en arquitectura de autoencoder.
Usa el encoder del autoencoder y agrega capas de clasificación.
"""

import torch
import torch.nn as nn


class ConvClassifier(nn.Module):
    """
    Clasificador binario basado en arquitectura de autoencoder.
    Usa el encoder del autoencoder y agrega capas de clasificación.
    """
    
    def __init__(self, in_channels: int = 3, feature_dims: int = 64, num_classes: int = 2):
        """
        Args:
            in_channels: Número de canales de entrada (3 para RGB)
            feature_dims: Dimensión del espacio latente (número de canales en el bottleneck)
            num_classes: Número de clases (2 para binario: normal/fallas)
        """
        super(ConvClassifier, self).__init__()
        
        # Encoder (mismo que el autoencoder)
        # 256x256 -> 128x128
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 256 -> 128
        )
        
        # 128x128 -> 64x64
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 128 -> 64
        )
        
        # 64x64 -> 32x32
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, feature_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 64 -> 32
        )
        
        # Clasificador
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Capas fully connected
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del clasificador.
        
        Args:
            x: Tensor de entrada de forma [batch, C, H, W]
        
        Returns:
            Tensor de logits de forma [batch, num_classes]
        """
        # Encoder
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        
        # Global Average Pooling
        x = self.global_pool(x)  # [batch, feature_dims, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, feature_dims]
        
        # Clasificador
        x = self.classifier(x)
        
        return x


class ResNetClassifier(nn.Module):
    """
    Clasificador binario basado en ResNet con transfer learning.
    Usa un encoder ResNet preentrenado y agrega capas de clasificación.
    """
    
    def __init__(
        self,
        encoder_name: str = 'resnet18',
        in_channels: int = 3,
        num_classes: int = 2,
        freeze_encoder: bool = False,
        pretrained: bool = True
    ):
        """
        Args:
            encoder_name: Nombre del modelo preentrenado ('resnet18', 'resnet34', 'resnet50')
            in_channels: Número de canales de entrada (3 para RGB)
            num_classes: Número de clases (2 para binario: normal/fallas)
            freeze_encoder: Si True, congela los pesos del encoder
            pretrained: Si True, usa pesos preentrenados en ImageNet
        """
        super(ResNetClassifier, self).__init__()
        
        self.encoder_name = encoder_name
        self.freeze_encoder = freeze_encoder
        
        # Cargar modelo preentrenado
        if encoder_name == 'resnet18':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
            feature_dims = 512
        elif encoder_name == 'resnet34':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained)
            feature_dims = 512
        elif encoder_name == 'resnet50':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
            feature_dims = 2048
        else:
            raise ValueError(f"Encoder no soportado: {encoder_name}")
        
        # Remover la capa fully connected original
        self.encoder = nn.Sequential(*list(model.children())[:-1])
        
        # Ajustar primera capa si in_channels != 3
        if in_channels != 3:
            old_conv = self.encoder[0]
            self.encoder[0] = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
        
        # Congelar encoder si se solicita
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del clasificador.
        
        Args:
            x: Tensor de entrada de forma [batch, C, H, W]
        
        Returns:
            Tensor de logits de forma [batch, num_classes]
        """
        # Encoder (preentrenado)
        with torch.set_grad_enabled(not self.freeze_encoder):
            x = self.encoder(x)  # [batch, feature_dims, H', W']
            x = x.view(x.size(0), -1)  # [batch, feature_dims]
        
        # Clasificador
        x = self.classifier(x)
        
        return x
    
    def unfreeze_encoder(self):
        """Descongela el encoder para permitir fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False
    
    def freeze_encoder_weights(self):
        """Congela el encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True

