# TesisMDP Supervisado - Sistema de Clasificación Binaria

Sistema completo para entrenamiento y validación de modelos de clasificación binaria (normal/fallas) utilizando múltiples arquitecturas de deep learning. El sistema incluye 25 modelos diferentes organizados en 5 grupos principales, cada uno con múltiples variantes para comparación.

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Requisitos e Instalación](#requisitos-e-instalación)
4. [Configuración](#configuración)
5. [Estructura del Dataset](#estructura-del-dataset)
6. [Procesamiento de Imágenes](#procesamiento-de-imágenes)
7. [Modelos Disponibles](#modelos-disponibles)
8. [Uso del Sistema](#uso-del-sistema)
9. [Validación de Modelos](#validación-de-modelos)
10. [Características Implementadas](#características-implementadas)
11. [Parámetros y Opciones](#parámetros-y-opciones)
12. [Archivos Generados](#archivos-generados)
13. [Ejemplos de Uso](#ejemplos-de-uso)

---

## Descripción General

Este proyecto implementa un sistema completo de entrenamiento supervisado para clasificación binaria de imágenes. El sistema está diseñado para:

- **Entrenar múltiples arquitecturas** de forma automática y comparativa
- **Procesar imágenes en parches** de 224x224 sin reescalar las imágenes originales
- **Implementar early stopping** automático para evitar sobreentrenamiento
- **Generar métricas completas** y matrices de confusión
- **Validar modelos entrenados** en datasets de prueba

**Total de modelos entrenables: 25 variantes** distribuidas en 5 grupos principales.

---

## Estructura del Proyecto

```
TesisMDP_supervised/
├── config.py                          # Configuración centralizada
├── dataset_supervisado.py             # Dataset con división en parches
├── train_all_models.py                # Script principal para entrenar todos los modelos
├── validate_model.py                  # Script para validar modelos entrenados
├── requirements.txt                   # Dependencias del proyecto
│
├── modelos/
│   ├── modelo1_autoencoder/
│   │   ├── train_supervised.py        # Entrenamiento modelo 1
│   │   ├── model_classifier.py        # Clasificadores originales
│   │   ├── autoencoder_models.py      # 5 variantes de autoencoders
│   │   ├── models/                    # Modelos entrenados guardados aquí
│   │   └── outputs/                   # Resultados y métricas
│   │
│   ├── modelo2_features/
│   │   ├── train_supervised.py        # Entrenamiento modelo 2
│   │   ├── models/                    # Modelos entrenados
│   │   └── outputs/                   # Resultados
│   │
│   ├── modelo3_transformer/
│   │   ├── train_supervised.py        # Entrenamiento modelo 3 (ViT)
│   │   ├── models/                    # Modelos entrenados
│   │   └── outputs/                   # Resultados
│   │
│   ├── modelo4_fastflow/
│   │   ├── train_supervised.py        # Entrenamiento modelo 4
│   │   ├── models/                    # Modelos entrenados
│   │   └── outputs/                   # Resultados
│   │
│   └── modelo5_stpm/
│       ├── train_supervised.py        # Entrenamiento modelo 5
│       ├── models/                    # Modelos entrenados
│       └── outputs/                   # Resultados
│
├── preprocesamiento/
│   ├── preprocesamiento.py            # Funciones de preprocesamiento
│   └── correct_board.py               # Corrección de tableros
│
└── outputs/                           # Resultados globales de validación
```

---

## Requisitos e Instalación

### Requisitos del Sistema

- **Python**: 3.8 o superior
- **CUDA**: Recomendado para entrenamiento (GPU NVIDIA compatible)
- **RAM**: Mínimo 8GB (recomendado 16GB+)
- **Espacio en disco**: Depende del tamaño del dataset

### Instalación de Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```

### Dependencias Principales

- **PyTorch** >= 1.9.0 (con soporte CUDA recomendado)
- **Torchvision** >= 0.10.0
- **OpenCV** >= 4.5.0
- **scikit-learn** >= 1.0.0
- **matplotlib** >= 3.5.0
- **seaborn** >= 0.11.0
- **tensorboard** >= 2.8.0

---

## Configuración

### 1. Configurar Ruta del Dataset

Edita el archivo `config.py` y actualiza la ruta a tu dataset:

```python
DATASET_SUPERVISADO_PATH = r"E:\Dataset\Validacion_procesadas"  # Cambiar esta ruta
```

**Importante**: El dataset debe contener dos subcarpetas:
- `normal/` - Imágenes sin fallas (label=0)
- `fallas/` - Imágenes con fallas (label=1)

---

## Estructura del Dataset

El dataset debe tener la siguiente estructura:

```
dataset_supervisado/
├── normal/
│   ├── imagen1.png
│   ├── imagen2.png
│   ├── imagen3.jpg
│   └── ...
└── fallas/
    ├── imagen1.png
    ├── imagen2.png
    ├── imagen3.jpg
    └── ...
```

### Formatos Soportados

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tif, .tiff)

---

## Procesamiento de Imágenes

### Características del Procesamiento

1. **División en Parches**: Las imágenes se dividen en parches de **224x224 píxeles** sin reescalar la imagen original
2. **Sin Preprocesamiento**: Las imágenes se usan directamente
3. **Conversión de Canales**:
   - Imágenes a color (3 canales): Se convierten de BGR a RGB
   - Imágenes en escala de grises (1 canal): Se replica a 3 canales (R=G=B)
4. **Normalización**: Las imágenes se normalizan a [0, 1] antes de pasarlas al modelo

### División en Parches

- **Tamaño de parche**: 224x224 píxeles (por defecto)
- **Sin reescalado**: La imagen original mantiene su tamaño
- **Extracción**: Se extrae el primer parche de cada imagen (se puede extender para usar todos los parches)

---

## Modelos Disponibles

El sistema incluye **25 modelos diferentes** organizados en 5 grupos:

### **Modelo 1: Autoencoders** (5 variantes)

Basado en diferentes arquitecturas de autoencoders como encoder para el clasificador:

1. **ConvAE** (Convolutional Autoencoder)
   - Arquitectura convolucional básica
   - Feature dimensions: 64
   - Archivo: `classifier_convae.pt`

2. **U-Net Autoencoder**
   - Encoder basado en arquitectura U-Net
   - Feature dimensions: 64
   - Archivo: `classifier_unet.pt`

3. **VAE** (Variational Autoencoder)
   - Encoder variacional con espacio latente estocástico
   - Feature dimensions: 64
   - Archivo: `classifier_vae.pt`

4. **Denoising Autoencoder**
   - Encoder con regularización (BatchNorm y Dropout)
   - Feature dimensions: 64
   - Archivo: `classifier_denoising.pt`

5. **ResNet-18 Autoencoder**
   - Encoder basado en ResNet-18 preentrenado
   - Feature dimensions: 512
   - Archivo: `classifier_resnet18.pt`

---

### **Modelo 2: Backbones** (5 variantes)

Clasificador basado en diferentes backbones preentrenados:

1. **ResNet18**
   - Feature dimensions: 512
   - Archivo: `modelo2_resnet18.pt`

2. **WideResNet50-2**
   - Feature dimensions: 2048
   - Archivo: `modelo2_wide_resnet50_2.pt`

3. **EfficientNet-B0**
   - Feature dimensions: 1280
   - Archivo: `modelo2_efficientnet_b0.pt`

4. **VGG16**
   - Feature dimensions: 512
   - Archivo: `modelo2_vgg16.pt`

5. **DenseNet121**
   - Feature dimensions: 1024
   - Archivo: `modelo2_densenet121.pt`

---

### **Modelo 3: Vision Transformer (ViT)** (5 variantes)

Clasificador basado en Vision Transformers con diferentes configuraciones:

1. **ViT-B/16** (Base, patch size 16)
   - Feature dimensions: 768
   - Archivo: `modelo3_vit_b_16.pt`

2. **ViT-B/32** (Base, patch size 32)
   - Feature dimensions: 768
   - Archivo: `modelo3_vit_b_32.pt`

3. **ViT-L/16** (Large, patch size 16)
   - Feature dimensions: 1024
   - Archivo: `modelo3_vit_l_16.pt`

4. **ViT-L/32** (Large, patch size 32)
   - Feature dimensions: 1024
   - Archivo: `modelo3_vit_l_32.pt`

5. **ViT-H/14** (Huge, patch size 14)
   - Feature dimensions: 1280
   - Archivo: `modelo3_vit_h_14.pt`

**Clasificador del Modelo 3**: MLP de 3 capas (feature_dims → 512 → 256 → 2) con ReLU y Dropout(0.5)

---

### **Modelo 4: Backbones** (5 variantes)

Clasificador basado en diferentes backbones preentrenados:

1. **ResNet18**
   - Feature dimensions: 512
   - Archivo: `modelo4_resnet18.pt`

2. **WideResNet50-2**
   - Feature dimensions: 2048
   - Archivo: `modelo4_wide_resnet50_2.pt`

3. **EfficientNet-B0**
   - Feature dimensions: 1280
   - Archivo: `modelo4_efficientnet_b0.pt`

4. **VGG16**
   - Feature dimensions: 512
   - Archivo: `modelo4_vgg16.pt`

5. **DenseNet121**
   - Feature dimensions: 1024
   - Archivo: `modelo4_densenet121.pt`

---

### **Modelo 5: Backbones** (5 variantes)

Clasificador basado en diferentes backbones preentrenados:

1. **ResNet18**
   - Feature dimensions: 512
   - Archivo: `modelo5_resnet18.pt`

2. **WideResNet50-2**
   - Feature dimensions: 2048
   - Archivo: `modelo5_wide_resnet50_2.pt`

3. **EfficientNet-B0**
   - Feature dimensions: 1280
   - Archivo: `modelo5_efficientnet_b0.pt`

4. **VGG16**
   - Feature dimensions: 512
   - Archivo: `modelo5_vgg16.pt`

5. **DenseNet121**
   - Feature dimensions: 1024
   - Archivo: `modelo5_densenet121.pt`

---

## Uso del Sistema

### Entrenamiento de Modelos

#### Entrenar un Modelo Específico

```bash
# Entrenar Modelo 1 (entrena automáticamente las 5 variantes de autoencoders)
python train_all_models.py --modelo 1

# Entrenar Modelo 2 (entrena automáticamente las 5 variantes de backbones)
python train_all_models.py --modelo 2

# Entrenar Modelo 3 (entrena automáticamente las 5 variantes de ViT)
python train_all_models.py --modelo 3

# Entrenar Modelo 4 (entrena automáticamente las 5 variantes de backbones)
python train_all_models.py --modelo 4

# Entrenar Modelo 5 (entrena automáticamente las 5 variantes de backbones)
python train_all_models.py --modelo 5
```

#### Entrenar Todos los Modelos

```bash
# Entrenar todos los modelos (25 variantes en total)
python train_all_models.py --modelo all
```

#### Ejemplos con Parámetros Personalizados

```bash
# Entrenar con parámetros personalizados
python train_all_models.py --modelo 1 \
    --epochs 100 \
    --lr 0.0001 \
    --batch_size 64 \
    --img_size 224

# Entrenar sin early stopping
python train_all_models.py --modelo 2 --no-early_stopping

# Entrenar con patience personalizado
python train_all_models.py --modelo 3 --patience 15 --min_delta 0.0005
```

---

## Validación de Modelos

### Script de Validación

El script `validate_model.py` permite validar cualquier modelo entrenado en un dataset de validación.

#### Uso Básico

```bash
# Validar modelo 1 (ConvAE)
python validate_model.py \
    --modelo 1 \
    --model_path modelos/modelo1_autoencoder/models/classifier_convae.pt \
    --val_path ruta/al/dataset/validacion \
    --encoder_type convae

# Validar modelo 2 (ResNet18)
python validate_model.py \
    --modelo 2 \
    --model_path modelos/modelo2_features/models/modelo2_resnet18.pt \
    --val_path ruta/al/dataset/validacion \
    --backbone resnet18

# Validar modelo 3 (ViT-B/16)
python validate_model.py \
    --modelo 3 \
    --model_path modelos/modelo3_transformer/models/modelo3_vit_b_16.pt \
    --val_path ruta/al/dataset/validacion \
    --model_name vit_b_16
```

#### Estructura del Dataset de Validación

El dataset de validación debe tener la misma estructura que el de entrenamiento:

```
ruta/al/dataset/validacion/
├── normal/
│   └── ...
└── fallas/
    └── ...
```

### Salidas de la Validación

El script genera:

1. **Métricas en consola**:
   - Loss
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Matriz de confusión
   - Reporte de clasificación

2. **Archivo JSON** (`outputs/validation_results_modeloX_TIMESTAMP.json`):
   - Todas las métricas en formato JSON
   - Configuración usada
   - Timestamp

3. **Gráfica de Matriz de Confusión** (`outputs/confusion_matrix_modeloX_TIMESTAMP.png`):
   - Visualización de la matriz de confusión

---

## Características Implementadas

### 1. Early Stopping Automático

- **Activado por defecto** en todos los modelos
- **Parámetros por defecto**:
  - `patience`: 10 épocas
  - `min_delta`: 0.001 (para modelo 1 y 3) / 0.0001 (para modelo 4)
- **Detención automática** si la loss de validación < 0.001
- Se puede desactivar con `--no-early_stopping`

### 2. División en Parches

- **Tamaño de parche**: 224x224 píxeles (por defecto)
- **Sin reescalado**: Las imágenes originales no se reescalan
- **Extracción**: Se divide la imagen en parches y se usa el primer parche

### 3. Batch Size Automático

El sistema calcula automáticamente el batch size óptimo según la memoria GPU disponible:

- **GPU >= 24GB**: Batch sizes grandes (64-128)
- **GPU >= 12GB**: Batch sizes medianos (32-64)
- **GPU >= 8GB**: Batch sizes pequeños (16-32)
- **GPU < 8GB**: Batch sizes mínimos (4-16)

### 4. Split Automático de Datos

- **85%** para entrenamiento
- **15%** para validación
- **Proporción de clases mantenida** en ambos splits

### 5. Logging con TensorBoard

- Métricas guardadas automáticamente en `runs/`
- Incluye: Loss, Accuracy, Precision, Recall, F1-Score, Learning Rate

---

## Parámetros y Opciones

### Parámetros Globales

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--modelo` | str | - | Modelo a entrenar: `1`, `2`, `3`, `4`, `5`, `all` |
| `--data_dir` | str | None | Directorio con carpetas normal/fallas (usa config.py si no se especifica) |
| `--img_size` | int | 224 | Tamaño de los parches (224x224) |
| `--batch_size` | int | None | Tamaño de batch (se calcula automáticamente si no se especifica) |
| `--epochs` | int | 50 | Número de épocas |
| `--lr` | float | 1e-3 | Learning rate |
| `--early_stopping` | flag | True | Activar early stopping (por defecto activado) |
| `--no-early_stopping` | flag | - | Desactivar early stopping |
| `--patience` | int | 10 | Paciencia para early stopping (épocas sin mejora) |
| `--min_delta` | float | 0.001 | Mejora mínima para considerar mejora |
| `--forzar_cpu` | flag | False | Forzar uso de CPU (no recomendado) |

### Parámetros Específicos por Modelo

#### Modelo 1
- `--encoder_type`: Tipo de autoencoder (`convae`, `unet`, `vae`, `denoising`, `resnet18`)

#### Modelo 2, 4, 5
- `--backbone`: Backbone a usar (se entrenan todas las variantes automáticamente)

#### Modelo 3
- `--model_name`: Modelo ViT (`vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`, `vit_h_14`)

---

## Archivos Generados

### Durante el Entrenamiento

Cada modelo genera:

1. **Modelo entrenado** (`.pt`):
   - Guardado en `modelos/modeloX/models/`
   - Nombre según la variante entrenada

2. **Historial de entrenamiento** (`.json`):
   - Guardado en `modelos/modeloX/models/`
   - Contiene: loss, accuracy, precision, recall, f1 por época
   - Configuración usada

3. **Logs de TensorBoard**:
   - Guardados en `runs/`
   - Visualizables con: `tensorboard --logdir runs`

### Durante la Validación

1. **Resultados JSON** (`outputs/validation_results_modeloX_TIMESTAMP.json`)
2. **Matriz de confusión** (`outputs/confusion_matrix_modeloX_TIMESTAMP.png`)

---

## Ejemplos de Uso

### Ejemplo 1: Entrenar Todos los Modelos

```bash
# Entrenar todos los modelos con configuración por defecto
python train_all_models.py --modelo all
```

Esto entrenará:
- Modelo 1: 5 variantes de autoencoders
- Modelo 2: 5 variantes de backbones
- Modelo 3: 5 variantes de ViT
- Modelo 4: 5 variantes de backbones
- Modelo 5: 5 variantes de backbones

**Total: 25 modelos**

### Ejemplo 2: Entrenar Solo Modelo 1 con Configuración Personalizada

```bash
python train_all_models.py --modelo 1 \
    --epochs 100 \
    --lr 0.0001 \
    --batch_size 32 \
    --patience 15
```

### Ejemplo 3: Validar un Modelo Específico

```bash
# Validar ViT-B/16
python validate_model.py \
    --modelo 3 \
    --model_path modelos/modelo3_transformer/models/modelo3_vit_b_16.pt \
    --val_path E:\Dataset\Validacion_procesadas \
    --model_name vit_b_16 \
    --batch_size 32
```

### Ejemplo 4: Entrenar sin Early Stopping

```bash
python train_all_models.py --modelo 2 --no-early_stopping --epochs 100
```

### Ejemplo 5: Entrenar con Tamaño de Parche Personalizado

```bash
python train_all_models.py --modelo 1 --img_size 256
```

---

## Resumen de Modelos por Grupo

| Grupo | Variantes | Arquitecturas |
|-------|-----------|---------------|
| **Modelo 1** | 5 | ConvAE, U-Net, VAE, Denoising, ResNet-18 AE |
| **Modelo 2** | 5 | ResNet18, WideResNet50-2, EfficientNet-B0, VGG16, DenseNet121 |
| **Modelo 3** | 5 | ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32, ViT-H/14 |
| **Modelo 4** | 5 | ResNet18, WideResNet50-2, EfficientNet-B0, VGG16, DenseNet121 |
| **Modelo 5** | 5 | ResNet18, WideResNet50-2, EfficientNet-B0, VGG16, DenseNet121 |
| **TOTAL** | **25** | - |

---

## Notas Importantes

1. **GPU Recomendada**: El entrenamiento es mucho más rápido con GPU. El sistema detecta automáticamente la GPU disponible.

2. **Memoria**: Los modelos más grandes (ViT-H/14, WideResNet50-2) requieren más memoria GPU.

3. **Tiempo de Entrenamiento**: Entrenar todos los modelos puede tomar varias horas dependiendo del hardware y tamaño del dataset.

4. **Early Stopping**: Está activado por defecto para evitar sobreentrenamiento y ahorrar tiempo.

5. **Parches**: El sistema divide las imágenes en parches de 224x224. Si una imagen es más pequeña que 224x224, se reescalará automáticamente.

---

## Solución de Problemas

### Error: "CUDA no está disponible"
- Verifica que tengas una GPU NVIDIA compatible
- Instala PyTorch con soporte CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Error: "No se encuentra el dataset"
- Verifica la ruta en `config.py`
- Asegúrate de que existan las carpetas `normal/` y `fallas/`

### Error: "Out of Memory"
- Reduce el `batch_size` manualmente
- Usa modelos más pequeños (ResNet18 en lugar de WideResNet50-2)

### Warning: "A single label was found"
- Esto ocurre cuando todas las predicciones son de una sola clase
- Normalmente se resuelve con más épocas de entrenamiento

---

## Referencias

- **PyTorch**: https://pytorch.org/
- **Torchvision Models**: https://pytorch.org/vision/stable/models.html
- **Vision Transformer**: https://arxiv.org/abs/2010.11929
- **EfficientNet**: https://arxiv.org/abs/1905.11946

---

## Autor

Proyecto desarrollado para Tesis de Maestría en Data Science.

---

## Licencia

Este proyecto es parte de una investigación académica.

---

**Última actualización**: Diciembre 2024
