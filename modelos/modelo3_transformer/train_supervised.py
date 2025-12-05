"""
Script de entrenamiento supervisado para modelo 3: Clasificador basado en Vision Transformer.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

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

import config
from dataset_supervisado import SupervisedDataset


class ViTClassifier(nn.Module):
    """Clasificador binario basado en Vision Transformer."""
    
    def __init__(self, model_name: str = 'vit_b_16', num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        
        # Cargar ViT preentrenado
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
            # ViT-H/14 puede no estar disponible en todas las versiones
            # Intentar cargar, si falla usar una alternativa
            try:
                from torchvision.models import vit_h_14
                self.backbone = vit_h_14(pretrained=pretrained)
                feature_dims = 1280
            except (ImportError, AttributeError):
                # Si no está disponible, usar ViT-L/16 como alternativa
                print("ADVERTENCIA: vit_h_14 no está disponible, usando vit_l_16 como alternativa")
                from torchvision.models import vit_l_16
                self.backbone = vit_l_16(pretrained=pretrained)
                feature_dims = 1024
        else:
            raise ValueError(f"Modelo ViT no soportado: {model_name}")
        
        # Reemplazar el clasificador final
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


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(train_loader), 100 * correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    return running_loss / len(val_loader), 100 * correct / total, precision, recall, f1, cm


def main():
    parser = argparse.ArgumentParser(description='Entrenar clasificador supervisado modelo 3')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--img_size', type=int, default=224)  # ViT usa 224x224
    parser.add_argument('--batch_size', type=int, default=16)  # ViT requiere más memoria
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_split', type=float, default=0.85)
    parser.add_argument('--model_name', type=str, default='vit_b_16', choices=['vit_b_16', 'vit_b_32', 'vit_l_16'])
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.001)
    
    args = parser.parse_args()
    
    data_dir = args.data_dir if args.data_dir else config.obtener_ruta_dataset_supervisado()
    output_dir = args.output_dir if args.output_dir else "models"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_dataset = SupervisedDataset(data_dir, 'train', args.img_size, train_split=args.train_split)
    val_dataset = SupervisedDataset(data_dir, 'val', args.img_size, train_split=args.train_split)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = ViTClassifier(model_name=args.model_name, num_classes=2).to(device)
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"modelo3_{args.model_name}_{timestamp}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}") if TENSORBOARD_AVAILABLE else None
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'epoch': []}
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*70}\nÉPOCA {epoch}/{args.epochs}\n{'='*70}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, precision, recall, f1, cm = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['val_precision'].append(float(precision))
        history['val_recall'].append(float(recall))
        history['val_f1'].append(float(f1))
        history['epoch'].append(epoch)
        
        print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
        
        # Detener si la loss es muy baja (convergencia completa)
        if val_loss < 0.001:
            print(f"\nEntrenamiento detenido: Loss de validación muy baja ({val_loss:.6f} < 0.001)")
            break
        
        if val_acc > best_val_acc:
            mejora = val_acc - best_val_acc
            if not args.early_stopping or mejora >= args.min_delta:
                best_val_acc = val_acc
                patience_counter = 0
                model_path = os.path.join(output_dir, f"modelo3_{args.model_name}.pt")
                torch.save(model.state_dict(), model_path)
                print(f"✓ Mejor modelo guardado: {model_path}")
            else:
                patience_counter += 1
        else:
            patience_counter += 1
        
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\nEarly stopping: No hay mejora en {args.patience} épocas")
            break
    
    history_path = os.path.join(output_dir, f"training_history_{run_name}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    if writer:
        writer.close()
    
    print(f"\n{'='*70}\nENTRENAMIENTO COMPLETADO\n{'='*70}")
    print(f"Mejor accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()

