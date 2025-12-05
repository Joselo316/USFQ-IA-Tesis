"""
Dataset supervisado para clasificación binaria (normal/fallas).
Carga imágenes desde carpetas 'normal' y 'fallas' con split 85/15 para validación.

IMPORTANTE: Usa las imágenes directamente sin preprocesamiento.
Si la imagen es a color (3 canales), se usa tal cual.
Si la imagen es en escala de grises (1 canal), se convierte a 3 canales replicando el canal.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, random_split
import cv2
import numpy as np

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "preprocesamiento") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "preprocesamiento"))

import config
# Nota: No se importa preprocesar_imagen_3canales porque usamos las imágenes directamente


class SupervisedDataset(Dataset):
    """
    Dataset supervisado para clasificación binaria.
    Carga imágenes desde carpetas 'normal' (label=0) y 'fallas' (label=1).
    """
    
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # 'train' o 'val'
        img_size: int = 224,
        transform: Optional[callable] = None,
        train_split: float = 0.85,
        random_seed: int = 42,
        use_patches: bool = True
    ):
        """
        Args:
            data_dir: Directorio raíz que contiene carpetas 'normal' y 'fallas'
            split: 'train' o 'val' para dividir el dataset
            img_size: Tamaño de los parches en los que se divide la imagen (default: 224)
            transform: Transformaciones adicionales (opcional)
            train_split: Ratio de datos para entrenamiento (default: 0.85)
            random_seed: Semilla para reproducibilidad del split
            use_patches: Si True, divide las imágenes en parches sin reescalar (default: True)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.use_patches = use_patches
        
        self.image_paths: List[Path] = []
        self.labels: List[int] = []  # 0 = normal, 1 = fallas
        self.patch_info: List[Tuple[int, int]] = []  # (patch_idx, total_patches) para cada entrada
        
        # Cargar todas las imágenes
        self._load_images()
        
        if len(self.image_paths) == 0:
            raise ValueError(
                f"No se encontraron imágenes en {data_dir}. "
                f"Asegúrate de que existan carpetas 'normal' y 'fallas' con imágenes válidas."
            )
        
        # Dividir en train/val si es necesario
        if split in ['train', 'val']:
            self._split_dataset(train_split, random_seed)
        # Si split es 'all', usar todas las imágenes sin dividir
        
        print(f"Dataset {split}: {len(self.image_paths)} {'parches' if use_patches else 'imágenes'}")
        print(f"  - Normal: {sum(1 for l in self.labels if l == 0)}")
        print(f"  - Fallas: {sum(1 for l in self.labels if l == 1)}")
    
    def _load_images(self):
        """Carga las rutas de imágenes desde las carpetas 'normal' y 'fallas'."""
        normal_dir = self.data_dir / 'normal'
        fallas_dir = self.data_dir / 'fallas'
        
        # Cargar imágenes normales (label=0)
        if normal_dir.exists():
            for ext in self.IMAGE_EXTENSIONS:
                for img_path in normal_dir.glob(f"*{ext}"):
                    self.image_paths.append(img_path)
                    self.labels.append(0)
                for img_path in normal_dir.glob(f"*{ext.upper()}"):
                    self.image_paths.append(img_path)
                    self.labels.append(0)
        else:
            print(f"ADVERTENCIA: No se encontró la carpeta 'normal' en {self.data_dir}")
        
        # Cargar imágenes con fallas (label=1)
        if fallas_dir.exists():
            for ext in self.IMAGE_EXTENSIONS:
                for img_path in fallas_dir.glob(f"*{ext}"):
                    self.image_paths.append(img_path)
                    self.labels.append(1)
                for img_path in fallas_dir.glob(f"*{ext.upper()}"):
                    self.image_paths.append(img_path)
                    self.labels.append(1)
        else:
            print(f"ADVERTENCIA: No se encontró la carpeta 'fallas' en {self.data_dir}")
    
    def _split_dataset(self, train_split: float, random_seed: int):
        """Divide el dataset en train/val manteniendo la proporción de clases."""
        # Crear índices para cada clase
        normal_indices = [i for i, label in enumerate(self.labels) if label == 0]
        fallas_indices = [i for i, label in enumerate(self.labels) if label == 1]
        
        # Dividir cada clase por separado
        normal_train_size = int(len(normal_indices) * train_split)
        fallas_train_size = int(len(fallas_indices) * train_split)
        
        generator = torch.Generator().manual_seed(random_seed)
        
        if len(normal_indices) > 0:
            normal_train, normal_val = random_split(
                normal_indices, 
                [normal_train_size, len(normal_indices) - normal_train_size],
                generator=generator
            )
        else:
            normal_train, normal_val = [], []
        
        if len(fallas_indices) > 0:
            fallas_train, fallas_val = random_split(
                fallas_indices,
                [fallas_train_size, len(fallas_indices) - fallas_train_size],
                generator=generator
            )
        else:
            fallas_train, fallas_val = [], []
        
        # Seleccionar índices según el split
        if self.split == 'train':
            selected_indices = list(normal_train) + list(fallas_train)
        else:  # val
            selected_indices = list(normal_val) + list(fallas_val)
        
        # Reordenar aleatoriamente pero mantener la semilla
        np.random.seed(random_seed)
        np.random.shuffle(selected_indices)
        
        # Filtrar paths y labels
        self.image_paths = [self.image_paths[i] for i in selected_indices]
        self.labels = [self.labels[i] for i in selected_indices]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _extract_patches(self, img: np.ndarray, patch_size: int) -> List[np.ndarray]:
        """
        Divide una imagen en parches de tamaño patch_size x patch_size.
        
        Args:
            img: Imagen de entrada (H, W, C)
            patch_size: Tamaño de cada parche
            
        Returns:
            Lista de parches
        """
        h, w = img.shape[:2]
        patches = []
        
        # Calcular número de parches
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        
        # Extraer parches
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y_start = i * patch_size
                y_end = y_start + patch_size
                x_start = j * patch_size
                x_end = x_start + patch_size
                
                patch = img[y_start:y_end, x_start:x_end]
                patches.append(patch)
        
        return patches
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retorna:
            image_tensor: Tensor de imagen (C, H, W) normalizado a [0, 1]
            label: 0 = normal, 1 = fallas
        
        NOTA: Si use_patches=True, divide la imagen en parches sin reescalar.
        - Si la imagen es a color (3 canales), se usa tal cual (convertida de BGR a RGB).
        - Si la imagen es en escala de grises (1 canal), se replica a 3 canales.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Intentar cargar como imagen a color (3 canales)
        img_color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        if img_color is not None:
            # Imagen a color: convertir BGR a RGB
            img_3ch = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        else:
            # Intentar cargar como escala de grises
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise ValueError(f"No se pudo cargar la imagen: {img_path}")
            
            # Replicar canal de grises a 3 canales (R=G=B)
            img_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        # Si use_patches=True, dividir en parches sin reescalar
        if self.use_patches:
            patches = self._extract_patches(img_3ch, self.img_size)
            if len(patches) == 0:
                # Si la imagen es más pequeña que el tamaño del parche, reescalar
                img_3ch = cv2.resize(img_3ch, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                patch = img_3ch
            else:
                # Usar el primer parche (o se puede hacer data augmentation con múltiples parches)
                # Por ahora usamos el primer parche
                patch = patches[0]
            
            img_3ch = patch
        else:
            # Redimensionar a tamaño objetivo si es necesario (comportamiento antiguo)
            if self.img_size is not None:
                h, w = img_3ch.shape[:2]
                if h != self.img_size or w != self.img_size:
                    img_3ch = cv2.resize(img_3ch, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convertir a tensor y normalizar a [0, 1]
        img_tensor = torch.from_numpy(img_3ch).float() / 255.0  # (H, W, 3)
        img_tensor = img_tensor.permute(2, 0, 1)  # (C, H, W)
        
        # Aplicar transformaciones adicionales si existen
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, label

