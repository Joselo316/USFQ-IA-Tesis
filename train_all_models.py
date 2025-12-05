"""
Script maestro para entrenar los 5 modelos de clasificación supervisada (normal/fallas).
Permite entrenar todos los modelos a la vez o seleccionar cuáles entrenar.
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime
import torch
import config

# Rutas a los scripts de entrenamiento supervisado
PROJECT_ROOT = Path(__file__).parent
TRAIN_MODEL1 = PROJECT_ROOT / "modelos" / "modelo1_autoencoder" / "train_supervised.py"
TRAIN_MODEL2 = PROJECT_ROOT / "modelos" / "modelo2_features" / "train_supervised.py"
TRAIN_MODEL3 = PROJECT_ROOT / "modelos" / "modelo3_transformer" / "train_supervised.py"
TRAIN_MODEL4 = PROJECT_ROOT / "modelos" / "modelo4_fastflow" / "train_supervised.py"
TRAIN_MODEL5 = PROJECT_ROOT / "modelos" / "modelo5_stpm" / "train_supervised.py"


def verificar_gpu():
    """Verifica que haya GPU disponible y muestra información."""
    if not torch.cuda.is_available():
        print("="*70)
        print("ERROR: CUDA no está disponible.")
        print("="*70)
        print("Este script requiere una GPU compatible con CUDA para entrenar los modelos.")
        print("Por favor, verifica que:")
        print("  1. Tienes una GPU NVIDIA compatible")
        print("  2. Tienes CUDA instalado correctamente")
        print("  3. PyTorch está compilado con soporte CUDA")
        print("="*70)
        return False
    
    print("="*70)
    print("VERIFICACIÓN DE GPU")
    print("="*70)
    print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA versión: {torch.version.cuda}")
    print(f"Número de GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 1:
        print(f"ADVERTENCIA: Se detectaron {torch.cuda.device_count()} GPUs, pero se usará solo la GPU 0")
    print("="*70)
    return True


def calcular_batch_size_optimo(memoria_gb: float, modelo: str) -> int:
    """
    Calcula un batch_size óptimo basado en la memoria GPU disponible.
    
    Args:
        memoria_gb: Memoria GPU en GB
        modelo: Nombre del modelo ('modelo1', 'modelo2', etc.)
    
    Returns:
        Batch size recomendado
    """
    # Batch sizes base según modelo y memoria
    if memoria_gb >= 24:  # GPU de alta gama (RTX 3090, A100, etc.)
        batch_sizes = {
            'modelo1': 128,
            'modelo2': 64,
            'modelo3': 64,
            'modelo4': 32,
            'modelo5': 32
        }
    elif memoria_gb >= 12:  # GPU media-alta (RTX 3080, RTX 4070, etc.)
        batch_sizes = {
            'modelo1': 64,
            'modelo2': 32,
            'modelo3': 32,
            'modelo4': 16,
            'modelo5': 16
        }
    elif memoria_gb >= 8:  # GPU media (RTX 3070, RTX 4060, etc.)
        batch_sizes = {
            'modelo1': 32,
            'modelo2': 16,
            'modelo3': 16,
            'modelo4': 8,
            'modelo5': 8
        }
    else:  # GPU baja (GTX 1660, RTX 3050, etc.)
        batch_sizes = {
            'modelo1': 16,
            'modelo2': 8,
            'modelo3': 8,
            'modelo4': 4,
            'modelo5': 4
        }
    
    return batch_sizes.get(modelo, 32)


def calcular_num_workers() -> int:
    """Calcula número óptimo de workers para DataLoader."""
    cpu_count = os.cpu_count() or 1
    return min(16, max(4, cpu_count // 2))


def entrenar_modelo1(args):
    """Entrena el modelo 1: Clasificador supervisado (5 variantes de autoencoders para comparación)"""
    print("\n" + "="*70)
    print("ENTRENANDO MODELO 1: CLASIFICADOR SUPERVISADO")
    print("="*70)
    print("Se entrenarán 5 variantes de autoencoders para comparación:")
    print("  1. ConvAE (Convolutional Autoencoder)")
    print("  2. U-Net Autoencoder")
    print("  3. VAE (Variational Autoencoder)")
    print("  4. Denoising Autoencoder")
    print("  5. ResNet-18 Autoencoder")
    print("="*70)
    
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset_supervisado()
    else:
        data_dir = args.data_dir
    
    if args.model1_output_dir:
        output_dir = args.model1_output_dir
    else:
        base_dir = Path(TRAIN_MODEL1).parent
        output_dir = str(base_dir / "models")
    
    if args.batch_size is None and torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size_modelo1 = calcular_batch_size_optimo(memoria_gb, 'modelo1')
    else:
        batch_size_modelo1 = args.batch_size or 32
    
    img_size = args.img_size if args.img_size is not None else 224
    
    resultados = []
    
    # Lista de variantes a entrenar
    variantes = [
        ('convae', 'ConvAE (Convolutional Autoencoder)'),
        ('unet', 'U-Net Autoencoder'),
        ('vae', 'VAE (Variational Autoencoder)'),
        ('denoising', 'Denoising Autoencoder'),
        ('resnet18', 'ResNet-18 Autoencoder')
    ]
    
    for idx, (encoder_type, nombre) in enumerate(variantes, 1):
        print("\n" + "="*70)
        print(f"VARIANTE {idx}/5: {nombre}")
        print("="*70)
        
        cmd = [
            sys.executable,
            str(TRAIN_MODEL1),
            "--data_dir", data_dir,
            "--img_size", str(img_size),
            "--batch_size", str(batch_size_modelo1),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--output_dir", output_dir,
            "--train_split", "0.85",
            "--encoder_type", encoder_type
        ]
        
        if hasattr(args, 'early_stopping') and args.early_stopping:
            cmd.append("--early_stopping")
            if hasattr(args, 'patience'):
                cmd.extend(["--patience", str(args.patience)])
            if hasattr(args, 'min_delta'):
                cmd.extend(["--min_delta", str(args.min_delta)])
        
        print(f"Comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        resultados.append((nombre, result.returncode == 0))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DEL ENTRENAMIENTO DEL MODELO 1")
    print("="*70)
    for nombre, exito in resultados:
        estado = "EXITOSO" if exito else "FALLIDO"
        print(f"  {nombre}: {estado}")
    print("="*70)
    
    # Retornar True si al menos uno fue exitoso
    return any(exito for _, exito in resultados)


def entrenar_modelo2(args):
    """Entrena el modelo 2: Clasificador supervisado (5 variantes de backbones)"""
    print("\n" + "="*70)
    print("ENTRENANDO MODELO 2: CLASIFICADOR SUPERVISADO")
    print("="*70)
    print("Se entrenarán 5 variantes de backbones para comparación:")
    print("  1. ResNet18")
    print("  2. WideResNet50-2")
    print("  3. EfficientNet-B0")
    print("  4. VGG16")
    print("  5. DenseNet121")
    print("="*70)
    
    if not TRAIN_MODEL2.exists():
        print(f"ERROR: No se encuentra el script de entrenamiento: {TRAIN_MODEL2}")
        return False
    
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset_supervisado()
    else:
        data_dir = args.data_dir
    
    if args.model2_output_dir:
        output_dir = args.model2_output_dir
    else:
        base_dir = Path(TRAIN_MODEL2).parent
        output_dir = str(base_dir / "models")
    
    if args.batch_size is None and torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size_modelo2 = calcular_batch_size_optimo(memoria_gb, 'modelo2')
    else:
        batch_size_modelo2 = args.batch_size or 32
    
    img_size = args.img_size if args.img_size is not None else 224
    
    resultados = []
    
    # Lista de variantes a entrenar
    variantes = [
        ('resnet18', 'ResNet18'),
        ('wide_resnet50_2', 'WideResNet50-2'),
        ('efficientnet_b0', 'EfficientNet-B0'),
        ('vgg16', 'VGG16'),
        ('densenet121', 'DenseNet121')
    ]
    
    for idx, (backbone, nombre) in enumerate(variantes, 1):
        print("\n" + "="*70)
        print(f"VARIANTE {idx}/5: {nombre}")
        print("="*70)
        
        cmd = [
            sys.executable,
            str(TRAIN_MODEL2),
            "--data_dir", data_dir,
            "--img_size", str(img_size),
            "--batch_size", str(batch_size_modelo2),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--backbone", backbone,
            "--output_dir", output_dir,
            "--train_split", "0.85"
        ]
        
        if args.early_stopping:
            cmd.append("--early_stopping")
            cmd.extend(["--patience", str(args.patience)])
            cmd.extend(["--min_delta", str(args.min_delta)])
        
        print(f"Comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        resultados.append((nombre, result.returncode == 0))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DEL ENTRENAMIENTO DEL MODELO 2")
    print("="*70)
    for nombre, exito in resultados:
        estado = "EXITOSO" if exito else "FALLIDO"
        print(f"  {nombre}: {estado}")
    print("="*70)
    
    # Retornar True si al menos uno fue exitoso
    return any(exito for _, exito in resultados)


def entrenar_modelo3(args):
    """Entrena el modelo 3: Clasificador supervisado basado en Vision Transformer (5 variantes)"""
    print("\n" + "="*70)
    print("ENTRENANDO MODELO 3: CLASIFICADOR SUPERVISADO (ViT)")
    print("="*70)
    print("Se entrenarán 5 variantes de Vision Transformer para comparación:")
    print("  1. ViT-B/16")
    print("  2. ViT-B/32")
    print("  3. ViT-L/16")
    print("  4. ViT-L/32")
    print("  5. ViT-H/14")
    print("="*70)
    
    if not TRAIN_MODEL3.exists():
        print(f"ERROR: No se encuentra el script de entrenamiento: {TRAIN_MODEL3}")
        return False
    
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset_supervisado()
    else:
        data_dir = args.data_dir
    
    if args.model3_output_dir:
        output_dir = args.model3_output_dir
    else:
        base_dir = Path(TRAIN_MODEL3).parent
        output_dir = str(base_dir / "models")
    
    if args.batch_size is None and torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size_modelo3 = calcular_batch_size_optimo(memoria_gb, 'modelo3')
    else:
        batch_size_modelo3 = args.batch_size or 16
    
    # ViT requiere imágenes de 224x224, no 256x256
    img_size = args.img_size if args.img_size is not None else 224
    
    resultados = []
    
    # Lista de variantes a entrenar
    variantes = [
        ('vit_b_16', 'ViT-B/16'),
        ('vit_b_32', 'ViT-B/32'),
        ('vit_l_16', 'ViT-L/16'),
        ('vit_l_32', 'ViT-L/32'),
        ('vit_h_14', 'ViT-H/14')
    ]
    
    for idx, (model_name, nombre) in enumerate(variantes, 1):
        print("\n" + "="*70)
        print(f"VARIANTE {idx}/5: {nombre}")
        print("="*70)
        
        cmd = [
            sys.executable,
            str(TRAIN_MODEL3),
            "--data_dir", data_dir,
            "--img_size", str(img_size),
            "--batch_size", str(batch_size_modelo3),
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--model_name", model_name,
            "--output_dir", output_dir,
            "--train_split", "0.85"
        ]
        
        if args.early_stopping:
            cmd.append("--early_stopping")
            cmd.extend(["--patience", str(args.patience)])
            cmd.extend(["--min_delta", str(args.min_delta)])
        
        print(f"Comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        resultados.append((nombre, result.returncode == 0))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DEL ENTRENAMIENTO DEL MODELO 3")
    print("="*70)
    for nombre, exito in resultados:
        estado = "EXITOSO" if exito else "FALLIDO"
        print(f"  {nombre}: {estado}")
    print("="*70)
    
    # Retornar True si al menos uno fue exitoso
    return any(exito for _, exito in resultados)


def entrenar_modelo4(args):
    """Entrena el modelo 4: Clasificador supervisado (5 variantes de backbones)"""
    print("\n" + "="*70)
    print("ENTRENANDO MODELO 4: CLASIFICADOR SUPERVISADO")
    print("="*70)
    print("Se entrenarán 5 variantes de backbones para comparación:")
    print("  1. ResNet18")
    print("  2. WideResNet50-2")
    print("  3. EfficientNet-B0")
    print("  4. VGG16")
    print("  5. DenseNet121")
    print("="*70)
    
    if not TRAIN_MODEL4.exists():
        print(f"ERROR: No se encuentra el script de entrenamiento: {TRAIN_MODEL4}")
        return False
    
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset_supervisado()
    else:
        data_dir = args.data_dir
    
    if args.model4_output_dir:
        output_dir = args.model4_output_dir
    else:
        base_dir = Path(TRAIN_MODEL4).parent
        output_dir = str(base_dir / "models")
    
    if args.batch_size is None and torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size_modelo4 = calcular_batch_size_optimo(memoria_gb, 'modelo4')
    else:
        batch_size_modelo4 = args.batch_size or 32
    
    img_size = args.img_size if args.img_size is not None else 224
    
    resultados = []
    
    # Lista de variantes a entrenar
    variantes = [
        ('resnet18', 'ResNet18'),
        ('wide_resnet50_2', 'WideResNet50-2'),
        ('efficientnet_b0', 'EfficientNet-B0'),
        ('vgg16', 'VGG16'),
        ('densenet121', 'DenseNet121')
    ]
    
    for idx, (backbone, nombre) in enumerate(variantes, 1):
        print("\n" + "="*70)
        print(f"VARIANTE {idx}/5: {nombre}")
        print("="*70)
        
        cmd = [
            sys.executable,
            str(TRAIN_MODEL4),
            "--data_dir", data_dir,
            "--img_size", str(img_size),
            "--batch_size", str(batch_size_modelo4),
            "--epochs", str(args.epochs),
            "--lr", str(args.model4_lr),
            "--backbone", backbone,
            "--output_dir", output_dir,
            "--train_split", "0.85"
        ]
        
        if args.model4_early_stopping:
            cmd.append("--early_stopping")
            cmd.extend(["--patience", str(args.model4_patience)])
            cmd.extend(["--min_delta", str(args.model4_min_delta)])
        
        print(f"Comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        resultados.append((nombre, result.returncode == 0))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DEL ENTRENAMIENTO DEL MODELO 4")
    print("="*70)
    for nombre, exito in resultados:
        estado = "EXITOSO" if exito else "FALLIDO"
        print(f"  {nombre}: {estado}")
    print("="*70)
    
    # Retornar True si al menos uno fue exitoso
    return any(exito for _, exito in resultados)


def entrenar_modelo5(args):
    """Entrena el modelo 5: Clasificador supervisado (5 variantes de backbones)"""
    print("\n" + "="*70)
    print("ENTRENANDO MODELO 5: CLASIFICADOR SUPERVISADO")
    print("="*70)
    print("Se entrenarán 5 variantes de backbones para comparación:")
    print("  1. ResNet18")
    print("  2. WideResNet50-2")
    print("  3. EfficientNet-B0")
    print("  4. VGG16")
    print("  5. DenseNet121")
    print("="*70)
    
    if not TRAIN_MODEL5.exists():
        print(f"ERROR: No se encuentra el script de entrenamiento: {TRAIN_MODEL5}")
        return False
    
    if args.data_dir is None:
        data_dir = config.obtener_ruta_dataset_supervisado()
    else:
        data_dir = args.data_dir
    
    if args.model5_output_dir:
        output_dir = args.model5_output_dir
    else:
        base_dir = Path(TRAIN_MODEL5).parent
        output_dir = str(base_dir / "models")
    
    if args.batch_size is None and torch.cuda.is_available() and not args.forzar_cpu:
        memoria_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size_modelo5 = calcular_batch_size_optimo(memoria_gb, 'modelo5')
    else:
        batch_size_modelo5 = args.batch_size or 32
    
    img_size = args.img_size if args.img_size is not None else 224
    
    resultados = []
    
    # Lista de variantes a entrenar
    variantes = [
        ('resnet18', 'ResNet18'),
        ('wide_resnet50_2', 'WideResNet50-2'),
        ('efficientnet_b0', 'EfficientNet-B0'),
        ('vgg16', 'VGG16'),
        ('densenet121', 'DenseNet121')
    ]
    
    for idx, (backbone, nombre) in enumerate(variantes, 1):
        print("\n" + "="*70)
        print(f"VARIANTE {idx}/5: {nombre}")
        print("="*70)
        
        cmd = [
            sys.executable,
            str(TRAIN_MODEL5),
            "--data_dir", data_dir,
            "--img_size", str(img_size),
            "--batch_size", str(batch_size_modelo5),
            "--epochs", str(args.epochs),
            "--lr", str(args.model5_lr),
            "--backbone", backbone,
            "--output_dir", output_dir,
            "--train_split", "0.85"
        ]
        
        if args.early_stopping:
            cmd.append("--early_stopping")
            cmd.extend(["--patience", str(args.patience)])
            cmd.extend(["--min_delta", str(args.min_delta)])
        
        print(f"Comando: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        resultados.append((nombre, result.returncode == 0))
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DEL ENTRENAMIENTO DEL MODELO 5")
    print("="*70)
    for nombre, exito in resultados:
        estado = "EXITOSO" if exito else "FALLIDO"
        print(f"  {nombre}: {estado}")
    print("="*70)
    
    # Retornar True si al menos uno fue exitoso
    return any(exito for _, exito in resultados)


def main():
    parser = argparse.ArgumentParser(
        description='Entrenar uno o todos los modelos de clasificación supervisada (normal/fallas)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Entrenar modelo 1 (entrena automáticamente 3 variantes para comparación:
  #   - ConvClassifier, ResNet18, ResNet50)
  python train_all_models.py --modelo 1

  # Entrenar modelo 2
  python train_all_models.py --modelo 2

  # Entrenar todos los modelos
  python train_all_models.py --modelo all

NOTA: El modelo 1 entrena automáticamente 5 variantes de autoencoders:
  1. ConvAE (Convolutional Autoencoder)
  2. U-Net Autoencoder
  3. VAE (Variational Autoencoder)
  4. Denoising Autoencoder
  5. ResNet-18 Autoencoder
Esto permite una comparación directa entre las diferentes arquitecturas de autoencoders.
        """
    )
    
    parser.add_argument(
        '--modelo',
        type=str,
        choices=['1', '2', '3', '4', '5', 'all', 'todos'],
        default=None,
        help='Modelo a entrenar: 1-5 o all/todos (todos los modelos)'
    )
    
    parser.add_argument('--all', action='store_true', help='Entrenar todos los modelos')
    parser.add_argument('--model1', action='store_true', help='Entrenar modelo 1')
    parser.add_argument('--model2', action='store_true', help='Entrenar modelo 2')
    parser.add_argument('--model3', action='store_true', help='Entrenar modelo 3')
    parser.add_argument('--model4', action='store_true', help='Entrenar modelo 4')
    parser.add_argument('--model5', action='store_true', help='Entrenar modelo 5')
    
    parser.add_argument('--data_dir', type=str, default=None, help='Directorio con carpetas normal/fallas')
    parser.add_argument('--batch_size', type=int, default=None, help='Tamaño de batch')
    parser.add_argument('--num_workers', type=int, default=None, help='Número de workers')
    parser.add_argument('--forzar_cpu', action='store_true', help='Forzar uso de CPU')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Tamaño de los parches en los que se divide la imagen (por defecto: 224x224)')
    
    parser.add_argument('--model1_transfer_learning', action='store_true', help='Usar transfer learning en modelo 1')
    parser.add_argument('--model1_encoder', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--model1_freeze_encoder', action='store_true', default=True)
    parser.add_argument('--model1_output_dir', type=str, default=None)
    
    parser.add_argument('--model2_backbone', type=str, default='wide_resnet50_2', 
                       choices=['resnet18', 'wide_resnet50_2', 'efficientnet_b0', 'vgg16', 'densenet121'],
                       help='Backbone para modelo 2 (DEPRECATED: ahora se entrenan todas las variantes)')
    parser.add_argument('--model2_output_dir', type=str, default=None)
    
    parser.add_argument('--model3_output_dir', type=str, default=None)
    
    parser.add_argument('--model4_backbone', type=str, default='resnet18', 
                       choices=['resnet18', 'wide_resnet50_2', 'efficientnet_b0', 'vgg16', 'densenet121'],
                       help='Backbone para modelo 4 (DEPRECATED: ahora se entrenan todas las variantes)')
    parser.add_argument('--model4_lr', type=float, default=1e-4)
    parser.add_argument('--model4_early_stopping', action='store_true', help='Usar early stopping en modelo 4 (por defecto: True)')
    parser.add_argument('--no-model4_early_stopping', dest='model4_early_stopping', action='store_false', help='Desactivar early stopping en modelo 4')
    parser.add_argument('--model4_patience', type=int, default=10)
    parser.add_argument('--model4_min_delta', type=float, default=0.0001)
    parser.add_argument('--model4_output_dir', type=str, default=None)
    
    parser.add_argument('--model5_backbone', type=str, default='resnet18', 
                       choices=['resnet18', 'wide_resnet50_2', 'efficientnet_b0', 'vgg16', 'densenet121'],
                       help='Backbone para modelo 5 (DEPRECATED: ahora se entrenan todas las variantes)')
    parser.add_argument('--model5_lr', type=float, default=1e-4)
    parser.add_argument('--model5_output_dir', type=str, default=None)
    
    parser.add_argument('--early_stopping', action='store_true', help='Usar early stopping (por defecto: True)')
    parser.add_argument('--no-early_stopping', dest='early_stopping', action='store_false', help='Desactivar early stopping')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=0.001)
    
    args = parser.parse_args()
    
    # Establecer early_stopping por defecto a True si no se especificó explícitamente
    if not hasattr(args, 'early_stopping'):
        args.early_stopping = True
    if not hasattr(args, 'model4_early_stopping'):
        args.model4_early_stopping = True
    
    if not args.forzar_cpu:
        if not verificar_gpu():
            print("\nERROR: No se puede continuar sin GPU.")
            print("Si realmente quieres usar CPU (NO recomendado), usa --forzar_cpu")
            return
    
    if args.modelo:
        if args.modelo in ['all', 'todos']:
            entrenar_modelo1_flag = True
            entrenar_modelo2_flag = True
            entrenar_modelo3_flag = True
            entrenar_modelo4_flag = True
            entrenar_modelo5_flag = True
        elif args.modelo == '1':
            entrenar_modelo1_flag = True
            entrenar_modelo2_flag = False
            entrenar_modelo3_flag = False
            entrenar_modelo4_flag = False
            entrenar_modelo5_flag = False
        elif args.modelo == '2':
            entrenar_modelo1_flag = False
            entrenar_modelo2_flag = True
            entrenar_modelo3_flag = False
            entrenar_modelo4_flag = False
            entrenar_modelo5_flag = False
        elif args.modelo == '3':
            entrenar_modelo1_flag = False
            entrenar_modelo2_flag = False
            entrenar_modelo3_flag = True
            entrenar_modelo4_flag = False
            entrenar_modelo5_flag = False
        elif args.modelo == '4':
            entrenar_modelo1_flag = False
            entrenar_modelo2_flag = False
            entrenar_modelo3_flag = False
            entrenar_modelo4_flag = True
            entrenar_modelo5_flag = False
        elif args.modelo == '5':
            entrenar_modelo1_flag = False
            entrenar_modelo2_flag = False
            entrenar_modelo3_flag = False
            entrenar_modelo4_flag = False
            entrenar_modelo5_flag = True
    elif args.all:
        entrenar_modelo1_flag = True
        entrenar_modelo2_flag = True
        entrenar_modelo3_flag = True
        entrenar_modelo4_flag = True
        entrenar_modelo5_flag = True
    else:
        entrenar_modelo1_flag = args.model1
        entrenar_modelo2_flag = args.model2
        entrenar_modelo3_flag = args.model3
        entrenar_modelo4_flag = args.model4
        entrenar_modelo5_flag = args.model5
    
    if not any([entrenar_modelo1_flag, entrenar_modelo2_flag, entrenar_modelo3_flag, 
                entrenar_modelo4_flag, entrenar_modelo5_flag]):
        parser.print_help()
        print("\nERROR: Debes especificar al menos un modelo para entrenar.")
        return
    
    print("="*70)
    print("ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN SUPERVISADA")
    print("="*70)
    if args.data_dir:
        print(f"Directorio de datos (especificado): {args.data_dir}")
    else:
        ruta_auto = config.obtener_ruta_dataset_supervisado()
        print(f"Directorio de datos (automático): {ruta_auto}")
        print(f"  - Dataset supervisado con carpetas: normal/ y fallas/")
        print(f"  - Split: 85% entrenamiento, 15% validación")
    print(f"Modelos a entrenar:")
    print(f"  - Modelo 1: {'Sí' if entrenar_modelo1_flag else 'No'}")
    print(f"  - Modelo 2: {'Sí' if entrenar_modelo2_flag else 'No'}")
    print(f"  - Modelo 3: {'Sí' if entrenar_modelo3_flag else 'No'}")
    print(f"  - Modelo 4: {'Sí' if entrenar_modelo4_flag else 'No'}")
    print(f"  - Modelo 5: {'Sí' if entrenar_modelo5_flag else 'No'}")
    print("="*70)
    
    inicio_total = time.time()
    resultados = {}
    
    if entrenar_modelo1_flag:
        inicio = time.time()
        exito = entrenar_modelo1(args)
        tiempo = time.time() - inicio
        resultados['modelo1'] = {'exito': exito, 'tiempo': tiempo}
    
    if entrenar_modelo2_flag:
        inicio = time.time()
        exito = entrenar_modelo2(args)
        tiempo = time.time() - inicio
        resultados['modelo2'] = {'exito': exito, 'tiempo': tiempo}
    
    if entrenar_modelo3_flag:
        inicio = time.time()
        exito = entrenar_modelo3(args)
        tiempo = time.time() - inicio
        resultados['modelo3'] = {'exito': exito, 'tiempo': tiempo}
    
    if entrenar_modelo4_flag:
        inicio = time.time()
        exito = entrenar_modelo4(args)
        tiempo = time.time() - inicio
        resultados['modelo4'] = {'exito': exito, 'tiempo': tiempo}
    
    if entrenar_modelo5_flag:
        inicio = time.time()
        exito = entrenar_modelo5(args)
        tiempo = time.time() - inicio
        resultados['modelo5'] = {'exito': exito, 'tiempo': tiempo}
    
    tiempo_total = time.time() - inicio_total
    print("\n" + "="*70)
    print("RESUMEN DEL ENTRENAMIENTO")
    print("="*70)
    for modelo, resultado in resultados.items():
        estado = "EXITOSO" if resultado['exito'] else "FALLIDO"
        tiempo_min = int(resultado['tiempo'] // 60)
        tiempo_sec = resultado['tiempo'] % 60
        print(f"{modelo.upper()}: {estado} - Tiempo: {tiempo_min} min {tiempo_sec:.1f} seg")
    print(f"\nTiempo total: {int(tiempo_total // 60)} min {tiempo_total % 60:.1f} seg")
    print("="*70)


if __name__ == "__main__":
    main()

