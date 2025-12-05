"""
Script para validar modelos de clasificación supervisada.
Permite evaluar un modelo entrenado en un dataset de validación y generar métricas.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Importar matplotlib y seaborn opcionales
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("ADVERTENCIA: matplotlib/seaborn no están instalados. No se generará la gráfica de matriz de confusión.")

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from dataset_supervisado import SupervisedDataset

# Importar modelos
from modelos.modelo1_autoencoder.model_classifier import ConvClassifier, ResNetClassifier as ResNetClassifier1
from modelos.modelo1_autoencoder.autoencoder_models import AutoencoderClassifier


class ResNetClassifier2(nn.Module):
    """Clasificador binario basado en diferentes backbones (Modelo 2)."""
    def __init__(self, backbone: str = 'wide_resnet50_2', num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        
        if backbone == 'resnet18':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
            feature_dims = 512
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'resnet50':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
            feature_dims = 2048
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'wide_resnet50_2':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=pretrained)
            feature_dims = 2048
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=pretrained)
            feature_dims = 1280
            self.encoder = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d(1)
            )
        elif backbone == 'vgg16':
            from torchvision.models import vgg16
            model = vgg16(pretrained=pretrained)
            feature_dims = 512
            self.encoder = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d(1)
            )
        elif backbone == 'densenet121':
            from torchvision.models import densenet121
            model = densenet121(pretrained=pretrained)
            feature_dims = 1024
            self.encoder = nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            raise ValueError(f"Backbone no soportado: {backbone}")
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ViTClassifier(nn.Module):
    """Clasificador binario basado en Vision Transformer (Modelo 3)."""
    def __init__(self, model_name: str = 'vit_b_16', num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        
        if model_name == 'vit_b_16':
            from torchvision.models import vit_b_16
            self.backbone = vit_b_16(pretrained=pretrained)
            feature_dims = 768
        elif model_name == 'vit_b_32':
            from torchvision.models import vit_b_32
            self.backbone = vit_b_32(pretrained=pretrained)
            feature_dims = 768
        elif model_name == 'vit_l_16':
            from torchvision.models import vit_l_16
            self.backbone = vit_l_16(pretrained=pretrained)
            feature_dims = 1024
        elif model_name == 'vit_l_32':
            from torchvision.models import vit_l_32
            self.backbone = vit_l_32(pretrained=pretrained)
            feature_dims = 1024
        elif model_name == 'vit_h_14':
            try:
                from torchvision.models import vit_h_14
                self.backbone = vit_h_14(pretrained=pretrained)
                feature_dims = 1280
            except (ImportError, AttributeError):
                print("ADVERTENCIA: vit_h_14 no está disponible, usando vit_l_16 como alternativa")
                from torchvision.models import vit_l_16
                self.backbone = vit_l_16(pretrained=pretrained)
                feature_dims = 1024
        else:
            raise ValueError(f"Modelo ViT no soportado: {model_name}")
        
        self.backbone.heads = nn.Sequential(
            nn.Linear(feature_dims, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class ResNetClassifier4(nn.Module):
    """Clasificador binario basado en diferentes backbones (Modelo 4)."""
    def __init__(self, backbone: str = 'resnet18', num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        
        if backbone == 'resnet18':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
            feature_dims = 512
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'resnet50':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
            feature_dims = 2048
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'wide_resnet50_2':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=pretrained)
            feature_dims = 2048
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=pretrained)
            feature_dims = 1280
            self.encoder = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d(1)
            )
        elif backbone == 'vgg16':
            from torchvision.models import vgg16
            model = vgg16(pretrained=pretrained)
            feature_dims = 512
            self.encoder = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d(1)
            )
        elif backbone == 'densenet121':
            from torchvision.models import densenet121
            model = densenet121(pretrained=pretrained)
            feature_dims = 1024
            self.encoder = nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            raise ValueError(f"Backbone no soportado: {backbone}")
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNetClassifier5(nn.Module):
    """Clasificador binario basado en diferentes backbones (Modelo 5)."""
    def __init__(self, backbone: str = 'resnet18', num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        
        if backbone == 'resnet18':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
            feature_dims = 512
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'resnet50':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
            feature_dims = 2048
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'wide_resnet50_2':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=pretrained)
            feature_dims = 2048
            self.encoder = nn.Sequential(*list(model.children())[:-1])
        elif backbone == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            model = efficientnet_b0(pretrained=pretrained)
            feature_dims = 1280
            self.encoder = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d(1)
            )
        elif backbone == 'vgg16':
            from torchvision.models import vgg16
            model = vgg16(pretrained=pretrained)
            feature_dims = 512
            self.encoder = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d(1)
            )
        elif backbone == 'densenet121':
            from torchvision.models import densenet121
            model = densenet121(pretrained=pretrained)
            feature_dims = 1024
            self.encoder = nn.Sequential(
                model.features,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            raise ValueError(f"Backbone no soportado: {backbone}")
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_model(modelo: int, model_path: str, device: str, **kwargs):
    """Carga un modelo entrenado."""
    print(f"\nCargando modelo {modelo} desde: {model_path}")
    
    if modelo == 1:
        # Modelo 1: Puede ser ConvClassifier, ResNetClassifier o AutoencoderClassifier
        encoder_type = kwargs.get('encoder_type', None)
        
        if encoder_type:
            # Usar nuevo sistema de autoencoders
            if encoder_type == 'resnet18':
                model = AutoencoderClassifier(
                    encoder_type='resnet18',
                    in_channels=3,
                    feature_dims=512,
                    num_classes=2
                )
            else:
                model = AutoencoderClassifier(
                    encoder_type=encoder_type,
                    in_channels=3,
                    feature_dims=64,
                    num_classes=2
                )
        elif 'use_transfer_learning' in kwargs and kwargs['use_transfer_learning']:
            # Compatibilidad hacia atrás
            encoder_name = kwargs.get('encoder_name', 'resnet18')
            model = ResNetClassifier1(
                encoder_name=encoder_name,
                in_channels=3,
                num_classes=2,
                freeze_encoder=False,
                pretrained=False  # Ya está entrenado
            )
        else:
            # ConvClassifier por defecto
            model = ConvClassifier(in_channels=3, feature_dims=64, num_classes=2)
    
    elif modelo == 2:
        backbone = kwargs.get('backbone', 'wide_resnet50_2')
        model = ResNetClassifier2(backbone=backbone, num_classes=2, pretrained=False)
    
    elif modelo == 3:
        model_name = kwargs.get('model_name', 'vit_b_16')
        model = ViTClassifier(model_name=model_name, num_classes=2, pretrained=False)
    
    elif modelo == 4:
        backbone = kwargs.get('backbone', 'resnet18')
        model = ResNetClassifier4(backbone=backbone, num_classes=2, pretrained=False)
    
    elif modelo == 5:
        backbone = kwargs.get('backbone', 'resnet18')
        model = ResNetClassifier5(backbone=backbone, num_classes=2, pretrained=False)
    
    else:
        raise ValueError(f"Modelo {modelo} no soportado")
    
    # Cargar pesos
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✓ Modelo cargado exitosamente")
    return model


def validate_model(model, val_loader, device):
    """Valida el modelo y retorna métricas."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    
    # Calcular métricas
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_confusion_matrix(cm, output_path=None):
    """Genera y guarda la matriz de confusión."""
    if not PLOTTING_AVAILABLE:
        print("ADVERTENCIA: No se puede generar la gráfica (matplotlib/seaborn no disponibles)")
        return
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fallas'],
                yticklabels=['Normal', 'Fallas'])
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusión guardada en: {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Validar un modelo de clasificación supervisada',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Validar modelo 1 (ConvClassifier)
  python validate_model.py --modelo 1 --model_path modelos/modelo1_autoencoder/models/classifier_conv.pt --val_path ruta/validacion

  # Validar modelo 2
  python validate_model.py --modelo 2 --model_path modelos/modelo2_features/models/modelo2_wide_resnet50_2.pt --val_path ruta/validacion --backbone wide_resnet50_2

  # Validar modelo 3
  python validate_model.py --modelo 3 --model_path modelos/modelo3_transformer/models/modelo3_vit_b_16.pt --val_path ruta/validacion --model_name vit_b_16
        """
    )
    
    parser.add_argument('--modelo', type=int, choices=[1, 2, 3, 4, 5], required=True,
                       help='Número del modelo a validar (1-5)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Ruta al archivo .pt del modelo entrenado')
    parser.add_argument('--val_path', type=str, required=True,
                       help='Ruta al directorio de validación (debe contener carpetas normal/ y fallas/)')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Tamaño de los parches en los que se divide la imagen (default: 256)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamaño de batch (default: 32)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directorio para guardar resultados (default: outputs/)')
    
    # Parámetros específicos por modelo
    parser.add_argument('--encoder_type', type=str, default=None,
                       choices=['convae', 'unet', 'vae', 'denoising', 'resnet18'],
                       help='Para modelo 1: tipo de autoencoder (convae, unet, vae, denoising, resnet18)')
    parser.add_argument('--use_transfer_learning', action='store_true',
                       help='Para modelo 1: usar transfer learning (DEPRECATED, usar --encoder_type)')
    parser.add_argument('--encoder_name', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Para modelo 1: nombre del encoder si usa transfer learning (DEPRECATED)')
    parser.add_argument('--backbone', type=str, default=None,
                       help='Para modelos 2, 4, 5: backbone (resnet18, resnet50, wide_resnet50_2)')
    parser.add_argument('--model_name', type=str, default='vit_b_16',
                       choices=['vit_b_16', 'vit_b_32', 'vit_l_16'],
                       help='Para modelo 3: nombre del modelo ViT')
    
    args = parser.parse_args()
    
    # Validar que el modelo existe
    if not os.path.exists(args.model_path):
        print(f"ERROR: No se encuentra el modelo en: {args.model_path}")
        return
    
    # Validar que el path de validación existe
    if not os.path.exists(args.val_path):
        print(f"ERROR: No se encuentra el directorio de validación en: {args.val_path}")
        return
    
    # Verificar estructura del dataset
    normal_dir = Path(args.val_path) / 'normal'
    fallas_dir = Path(args.val_path) / 'fallas'
    
    if not normal_dir.exists() or not fallas_dir.exists():
        print(f"ERROR: El directorio de validación debe contener carpetas 'normal' y 'fallas'")
        return
    
    # Configurar device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print("VALIDACIÓN DE MODELO")
    print(f"{'='*70}")
    print(f"Dispositivo: {device}")
    print(f"Modelo: {args.modelo}")
    print(f"Path de validación: {args.val_path}")
    print(f"{'='*70}")
    
    # Determinar parámetros del modelo según el número
    model_kwargs = {}
    if args.modelo == 1:
        if args.encoder_type:
            model_kwargs['encoder_type'] = args.encoder_type
        elif args.use_transfer_learning:
            model_kwargs['use_transfer_learning'] = True
            model_kwargs['encoder_name'] = args.encoder_name
    elif args.modelo == 2:
        model_kwargs['backbone'] = args.backbone or 'wide_resnet50_2'
    elif args.modelo == 3:
        model_kwargs['model_name'] = args.model_name
    elif args.modelo == 4:
        model_kwargs['backbone'] = args.backbone or 'resnet18'
    elif args.modelo == 5:
        model_kwargs['backbone'] = args.backbone or 'resnet18'
    
    # Cargar modelo
    model = load_model(args.modelo, args.model_path, device, **model_kwargs)
    
    # Crear dataset y dataloader
    print(f"\nCargando dataset de validación...")
    # Con train_split=0.0, todas las imágenes van a 'val', que es lo que queremos
    val_dataset = SupervisedDataset(
        data_dir=args.val_path,
        split='val',
        img_size=args.img_size,
        train_split=0.0  # Todas las imágenes van a validación
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"✓ Dataset cargado: {len(val_dataset)} imágenes")
    print(f"  - Normal: {sum(1 for l in val_dataset.labels if l == 0)}")
    print(f"  - Fallas: {sum(1 for l in val_dataset.labels if l == 1)}")
    
    # Validar modelo
    print(f"\n{'='*70}")
    print("EVALUANDO MODELO...")
    print(f"{'='*70}")
    
    results = validate_model(model, val_loader, device)
    
    # Mostrar resultados
    print(f"\n{'='*70}")
    print("RESULTADOS DE VALIDACIÓN")
    print(f"{'='*70}")
    print(f"Loss: {results['loss']:.6f}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"\nMatriz de Confusión:")
    print(f"                Predicción")
    print(f"              Normal  Fallas")
    print(f"Real Normal    {results['confusion_matrix'][0,0]:5d}   {results['confusion_matrix'][0,1]:5d}")
    print(f"     Fallas    {results['confusion_matrix'][1,0]:5d}   {results['confusion_matrix'][1,1]:5d}")
    print(f"{'='*70}")
    
    # Reporte de clasificación
    print(f"\nReporte de Clasificación:")
    print(classification_report(
        results['labels'],
        results['predictions'],
        target_names=['Normal', 'Fallas'],
        digits=4
    ))
    
    # Guardar resultados
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"validation_results_modelo{args.modelo}_{timestamp}.json"
    
    # Guardar métricas en JSON
    results_dict = {
        'modelo': args.modelo,
        'model_path': args.model_path,
        'val_path': args.val_path,
        'timestamp': timestamp,
        'metrics': {
            'loss': float(results['loss']),
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1']),
            'confusion_matrix': results['confusion_matrix'].tolist()
        },
        'config': {
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            **model_kwargs
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n✓ Resultados guardados en: {results_file}")
    
    # Guardar matriz de confusión
    cm_file = output_dir / f"confusion_matrix_modelo{args.modelo}_{timestamp}.png"
    plot_confusion_matrix(results['confusion_matrix'], str(cm_file))
    
    print(f"\n{'='*70}")
    print("VALIDACIÓN COMPLETADA")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

