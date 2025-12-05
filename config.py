"""
Archivo de configuración centralizado para el proyecto TesisMDP Supervisado.
Contiene la ruta al dataset supervisado con carpetas normal/fallas.
"""

import os
from pathlib import Path

# ============================================================================
# RUTA AL DATASET SUPERVISADO
# ============================================================================
# IMPORTANTE: Configurar la ruta absoluta al directorio donde están las imágenes del dataset supervisado.
# El dataset debe contener dos subcarpetas: 'normal' y 'fallas'.
# Ejemplo: DATASET_SUPERVISADO_PATH = r"D:\Dataset\supervisado"
DATASET_SUPERVISADO_PATH = r"E:\Dataset\Validacion_procesadas"  # CAMBIAR ESTA RUTA SEGÚN TU CONFIGURACIÓN

# Verificar que la ruta existe
if not os.path.exists(DATASET_SUPERVISADO_PATH):
    print(f"ADVERTENCIA: La ruta al dataset supervisado no existe: {DATASET_SUPERVISADO_PATH}")
    print("   Por favor, actualiza DATASET_SUPERVISADO_PATH en config.py con la ruta correcta.")
    print("   El dataset debe contener dos subcarpetas: 'normal' y 'fallas'.")

# Ratio de split para entrenamiento/validación (85% entrenamiento, 15% validación)
TRAIN_SPLIT_RATIO = 0.85
VAL_SPLIT_RATIO = 0.15

# ============================================================================
# PARÁMETROS COMUNES
# ============================================================================
# Tamaño de imagen objetivo para redimensionamiento
IMG_SIZE = 256

# ============================================================================
# RUTAS DE SALIDA
# ============================================================================
# Directorio base del proyecto
PROJECT_ROOT = Path(__file__).parent

# Directorios de salida para cada modelo
OUTPUT_DIR_MODEL1 = PROJECT_ROOT / "modelos" / "modelo1_autoencoder" / "outputs"
OUTPUT_DIR_MODEL2 = PROJECT_ROOT / "modelos" / "modelo2_features" / "outputs"
OUTPUT_DIR_MODEL3 = PROJECT_ROOT / "modelos" / "modelo3_transformer" / "outputs"
OUTPUT_DIR_MODEL4 = PROJECT_ROOT / "modelos" / "modelo4_fastflow" / "outputs"
OUTPUT_DIR_MODEL5 = PROJECT_ROOT / "modelos" / "modelo5_stpm" / "outputs"

# Crear directorios de salida si no existen
OUTPUT_DIR_MODEL1.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_MODEL2.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_MODEL3.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_MODEL4.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_MODEL5.mkdir(parents=True, exist_ok=True)


def obtener_ruta_dataset_supervisado() -> str:
    """
    Obtiene la ruta del dataset supervisado.
    
    Returns:
        Ruta al dataset supervisado con carpetas 'normal' y 'fallas'.
    """
    return DATASET_SUPERVISADO_PATH

