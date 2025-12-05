"""
Diferentes arquitecturas de autoencoders para el modelo 1.
Cada autoencoder se usa como encoder para el clasificador.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAEEncoder(nn.Module):
    """Encoder del Convolutional Autoencoder (ConvAE)."""
    def __init__(self, in_channels: int = 3, feature_dims: int = 64):
        super(ConvAEEncoder, self).__init__()
        # 256x256 -> 128x128
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # 128x128 -> 64x64
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # 64x64 -> 32x32
        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, feature_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        return x


class UNetEncoder(nn.Module):
    """Encoder basado en U-Net para autoencoder."""
    def __init__(self, in_channels: int = 3, feature_dims: int = 64):
        super(UNetEncoder, self).__init__()
        # Contraction path (encoder)
        self.enc1 = self._conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = self._conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = self._conv_block(512, feature_dims)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Guardar para skip connections (aunque no las usamos en el encoder)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bottleneck = self.bottleneck(self.pool4(e4))
        return bottleneck


class VAEEncoder(nn.Module):
    """Encoder del Variational Autoencoder (VAE)."""
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super(VAEEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder layers - usando convoluciones más simples
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling antes de las capas lineales
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Latent space
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        x = self.enc1(x)  # 256 -> 128
        x = self.enc2(x)  # 128 -> 64
        x = self.enc3(x)  # 64 -> 32
        x = self.enc4(x)  # 32 -> 16
        
        # Global Average Pooling
        x = self.global_pool(x)  # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 256]
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Para clasificación, usamos mu directamente (sin reparameterization trick)
        # ya que queremos una representación determinística
        z = mu.unsqueeze(-1).unsqueeze(-1)  # [batch, latent_dim, 1, 1]
        return z


class DenoisingAEEncoder(nn.Module):
    """Encoder del Denoising Autoencoder."""
    def __init__(self, in_channels: int = 3, feature_dims: int = 64):
        super(DenoisingAEEncoder, self).__init__()
        # Similar a ConvAE pero con más capas y dropout para regularización
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, 2)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv2d(128, feature_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dims),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        return x


class ResNet18AEEncoder(nn.Module):
    """Encoder basado en ResNet-18 para autoencoder."""
    def __init__(self, in_channels: int = 3, feature_dims: int = 512):
        super(ResNet18AEEncoder, self).__init__()
        # Cargar ResNet18 preentrenado
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
        # Usar todas las capas excepto la última fully connected
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
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
        
        # Proyección a feature_dims si es necesario
        if feature_dims != 512:
            self.projection = nn.Conv2d(512, feature_dims, kernel_size=1)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        return x


class AutoencoderClassifier(nn.Module):
    """
    Clasificador binario basado en diferentes arquitecturas de autoencoder.
    Usa el encoder del autoencoder y agrega capas de clasificación.
    """
    
    def __init__(
        self,
        encoder_type: str = 'convae',
        in_channels: int = 3,
        feature_dims: int = 64,
        num_classes: int = 2
    ):
        """
        Args:
            encoder_type: Tipo de encoder ('convae', 'unet', 'vae', 'denoising', 'resnet18')
            in_channels: Número de canales de entrada (3 para RGB)
            feature_dims: Dimensión del espacio latente
            num_classes: Número de clases (2 para binario: normal/fallas)
        """
        super(AutoencoderClassifier, self).__init__()
        
        self.encoder_type = encoder_type
        
        # Crear encoder según el tipo
        if encoder_type == 'convae':
            self.encoder = ConvAEEncoder(in_channels, feature_dims)
            actual_feature_dims = feature_dims
        elif encoder_type == 'unet':
            self.encoder = UNetEncoder(in_channels, feature_dims)
            actual_feature_dims = feature_dims
        elif encoder_type == 'vae':
            self.encoder = VAEEncoder(in_channels, feature_dims)
            actual_feature_dims = feature_dims
        elif encoder_type == 'denoising':
            self.encoder = DenoisingAEEncoder(in_channels, feature_dims)
            actual_feature_dims = feature_dims
        elif encoder_type == 'resnet18':
            self.encoder = ResNet18AEEncoder(in_channels, feature_dims)
            actual_feature_dims = feature_dims if feature_dims != 512 else 512
        else:
            raise ValueError(f"Tipo de encoder no soportado: {encoder_type}")
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(actual_feature_dims, 128),
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
        x = self.encoder(x)
        
        # Global Average Pooling
        x = self.global_pool(x)  # [batch, feature_dims, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, feature_dims]
        
        # Clasificador
        x = self.classifier(x)
        
        return x

