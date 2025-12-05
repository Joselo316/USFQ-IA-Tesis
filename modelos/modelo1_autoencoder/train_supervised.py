"""
Script de entrenamiento supervisado para clasificación binaria (normal/fallas).
Usa el dataset supervisado con split 85/15.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("ADVERTENCIA: TensorBoard no está instalado. Las métricas no se guardarán en TensorBoard.")

import config
from dataset_supervisado import SupervisedDataset
from modelos.modelo1_autoencoder.model_classifier import ConvClassifier, ResNetClassifier
from modelos.modelo1_autoencoder.autoencoder_models import AutoencoderClassifier


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Entrena el modelo por una época."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Estadísticas
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progreso
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={100*correct/total:.2f}%", end='\r')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"\n  Época {epoch}/{total_epochs} - Train Loss: {epoch_loss:.6f}, Train Acc: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Valida el modelo."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    # Calcular métricas adicionales
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    print(f"  Val Loss: {epoch_loss:.6f}, Val Acc: {epoch_acc:.2f}%")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    return epoch_loss, epoch_acc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar clasificador binario supervisado (normal/fallas)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help=f'Directorio raíz con carpetas normal/fallas (default: desde config.py)'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help=f'Tamaño de los parches en los que se divide la imagen (default: 224)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Tamaño del batch (default: 32)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Número de épocas (default: 50)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--train_split',
        type=float,
        default=0.85,
        help='Proporción de datos para entrenamiento (default: 0.85)'
    )
    parser.add_argument(
        '--encoder_type',
        type=str,
        default='convae',
        choices=['convae', 'unet', 'vae', 'denoising', 'resnet18'],
        help='Tipo de autoencoder: convae, unet, vae, denoising, resnet18 (default: convae)'
    )
    parser.add_argument(
        '--use_transfer_learning',
        action='store_true',
        help='Usar modelo con transfer learning (ResNet preentrenado) - DEPRECATED, usar --encoder_type resnet18'
    )
    parser.add_argument(
        '--encoder_name',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='Nombre del encoder cuando se usa transfer learning (default: resnet18) - DEPRECATED'
    )
    parser.add_argument(
        '--freeze_encoder',
        action='store_true',
        default=False,
        help='Congelar encoder cuando se usa transfer learning (default: False)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directorio para guardar modelo (default: models/)'
    )
    parser.add_argument(
        '--early_stopping',
        action='store_true',
        help='Activar early stopping (detener si no hay mejora)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Paciencia para early stopping (épocas sin mejora, default: 10)'
    )
    parser.add_argument(
        '--min_delta',
        type=float,
        default=0.001,
        help='Mejora mínima absoluta para considerar mejora (default: 0.001)'
    )
    
    args = parser.parse_args()
    
    # Obtener ruta del dataset
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset_supervisado()
    else:
        data_dir = args.data_dir
    
    output_dir = args.output_dir if args.output_dir else "models"
    
    # Dispositivo
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Dispositivo: {device} (GPU)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("ADVERTENCIA: CUDA no está disponible. Se usará CPU (entrenamiento será más lento).")
        device = "cpu"
    
    # Crear directorio para modelos
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar datasets
    print(f"\nCargando dataset desde {data_dir}...")
    train_dataset = SupervisedDataset(
        data_dir=data_dir,
        split='train',
        img_size=args.img_size,
        train_split=args.train_split
    )
    
    val_dataset = SupervisedDataset(
        data_dir=data_dir,
        split='val',
        img_size=args.img_size,
        train_split=args.train_split
    )
    
    # Crear DataLoaders
    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    
    # Crear modelo
    # Si se usa --use_transfer_learning (compatibilidad hacia atrás), usar ResNetClassifier
    if args.use_transfer_learning:
        print(f"\nCreando modelo con transfer learning (encoder: {args.encoder_name})...")
        model = ResNetClassifier(
            encoder_name=args.encoder_name,
            in_channels=3,
            num_classes=2,
            freeze_encoder=args.freeze_encoder,
            pretrained=True
        ).to(device)
        model_name = f"classifier_{args.encoder_name}.pt"
    else:
        # Usar el nuevo sistema de autoencoders
        encoder_type = args.encoder_type
        print(f"\nCreando modelo con autoencoder tipo: {encoder_type}")
        
        if encoder_type == 'resnet18':
            # Para ResNet18, usar feature_dims=512
            model = AutoencoderClassifier(
                encoder_type='resnet18',
                in_channels=3,
                feature_dims=512,
                num_classes=2
            ).to(device)
        else:
            # Para otros tipos, usar feature_dims=64
            model = AutoencoderClassifier(
                encoder_type=encoder_type,
                in_channels=3,
                feature_dims=64,
                num_classes=2
            ).to(device)
        
        model_name = f"classifier_{encoder_type}.pt"
    
    print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables: {trainable:,}")
    
    # Loss y optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"classifier_{timestamp}"
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=f"runs/{run_name}")
    else:
        writer = None
    
    # Entrenamiento
    print(f"\nIniciando entrenamiento por {args.epochs} épocas...")
    if args.early_stopping:
        print(f"Early stopping activado: paciencia={args.patience}, min_delta={args.min_delta}")
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'epoch': [],
        'learning_rate': [],
        'config': {
            'img_size': args.img_size,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'train_split': args.train_split,
            'use_transfer_learning': args.use_transfer_learning,
            'encoder_name': args.encoder_name if args.use_transfer_learning else None,
            'freeze_encoder': args.freeze_encoder if args.use_transfer_learning else None,
            'early_stopping': args.early_stopping,
            'patience': args.patience if args.early_stopping else None,
            'min_delta': args.min_delta if args.early_stopping else None
        }
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}")
        print(f"ÉPOCA {epoch}/{args.epochs}")
        print(f"{'='*70}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_acc, precision, recall, f1 = validate(model, val_loader, criterion, device)
        
        # Guardar historial
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['val_precision'].append(float(precision))
        history['val_recall'].append(float(recall))
        history['val_f1'].append(float(f1))
        history['epoch'].append(epoch)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            writer.add_scalar('Metrics/Precision', precision, epoch)
            writer.add_scalar('Metrics/Recall', recall, epoch)
            writer.add_scalar('Metrics/F1', f1, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Detener si la loss es muy baja (convergencia completa)
        if val_loss < 0.001:
            print(f"\nEntrenamiento detenido: Loss de validación muy baja ({val_loss:.6f} < 0.001)")
            break
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            mejora = val_acc - best_val_acc
            if not args.early_stopping or mejora >= args.min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(output_dir, model_name)
                torch.save(model.state_dict(), model_path)
                print(f"  ✓ Mejora detectada: {mejora:.4f}% (>= {args.min_delta:.4f}%)")
                print(f"  ✓ Mejor modelo guardado: {model_path}")
            else:
                print(f"  Mejora insuficiente: {mejora:.4f}% < {args.min_delta:.4f}%")
        else:
            patience_counter += 1
            if args.early_stopping and patience_counter >= args.patience:
                print(f"\nEarly stopping: No hay mejora en {args.patience} épocas")
                break
    
    # Guardar historial
    history_path = os.path.join(output_dir, f"training_history_{run_name}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nHistorial guardado: {history_path}")
    
    if writer is not None:
        writer.close()
    
    print(f"\n{'='*70}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Mejor accuracy de validación: {best_val_acc:.2f}%")
    print(f"Modelo guardado: {os.path.join(output_dir, model_name)}")


if __name__ == "__main__":
    main()

