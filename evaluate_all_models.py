"""
Script para evaluar todos los modelos entrenados con todas sus variantes.
Genera métricas completas y matrices de confusión para cada modelo.
"""

# Configurar encoding UTF-8 para Windows
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Agregar rutas al path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config


# Configuración de modelos y sus variantes
MODELOS_CONFIG = {
    1: {
        'nombre': 'Modelo 1: Autoencoders',
        'directorio': 'modelos/modelo1_autoencoder/models',
        'variantes': [
            ('classifier_convae.pt', 'convae', 'ConvAE'),
            ('classifier_unet.pt', 'unet', 'U-Net'),
            ('classifier_vae.pt', 'vae', 'VAE'),
            ('classifier_denoising.pt', 'denoising', 'Denoising'),
            ('classifier_resnet18.pt', 'resnet18', 'ResNet-18 AE'),
        ]
    },
    2: {
        'nombre': 'Modelo 2: Backbones',
        'directorio': 'modelos/modelo2_features/models',
        'variantes': [
            ('modelo2_resnet18.pt', 'resnet18', 'ResNet18'),
            ('modelo2_wide_resnet50_2.pt', 'wide_resnet50_2', 'WideResNet50-2'),
            ('modelo2_efficientnet_b0.pt', 'efficientnet_b0', 'EfficientNet-B0'),
            ('modelo2_vgg16.pt', 'vgg16', 'VGG16'),
            ('modelo2_densenet121.pt', 'densenet121', 'DenseNet121'),
        ]
    },
    3: {
        'nombre': 'Modelo 3: Vision Transformer',
        'directorio': 'modelos/modelo3_transformer/models',
        'variantes': [
            ('modelo3_vit_b_16.pt', 'vit_b_16', 'ViT-B/16'),
            ('modelo3_vit_b_32.pt', 'vit_b_32', 'ViT-B/32'),
            ('modelo3_vit_l_16.pt', 'vit_l_16', 'ViT-L/16'),
            ('modelo3_vit_l_32.pt', 'vit_l_32', 'ViT-L/32'),
            ('modelo3_vit_h_14.pt', 'vit_h_14', 'ViT-H/14'),
        ]
    },
    4: {
        'nombre': 'Modelo 4: Backbones (FastFlow)',
        'directorio': 'modelos/modelo4_fastflow/models',
        'variantes': [
            ('modelo4_resnet18.pt', 'resnet18', 'ResNet18'),
            ('modelo4_wide_resnet50_2.pt', 'wide_resnet50_2', 'WideResNet50-2'),
            ('modelo4_efficientnet_b0.pt', 'efficientnet_b0', 'EfficientNet-B0'),
            ('modelo4_vgg16.pt', 'vgg16', 'VGG16'),
            ('modelo4_densenet121.pt', 'densenet121', 'DenseNet121'),
        ]
    },
    5: {
        'nombre': 'Modelo 5: Backbones (STPM)',
        'directorio': 'modelos/modelo5_stpm/models',
        'variantes': [
            ('modelo5_resnet18.pt', 'resnet18', 'ResNet18'),
            ('modelo5_wide_resnet50_2.pt', 'wide_resnet50_2', 'WideResNet50-2'),
            ('modelo5_efficientnet_b0.pt', 'efficientnet_b0', 'EfficientNet-B0'),
            ('modelo5_vgg16.pt', 'vgg16', 'VGG16'),
            ('modelo5_densenet121.pt', 'densenet121', 'DenseNet121'),
        ]
    }
}


def encontrar_modelos_disponibles():
    """Encuentra todos los modelos entrenados disponibles."""
    modelos_encontrados = {}
    
    for modelo_num, config_modelo in MODELOS_CONFIG.items():
        directorio = PROJECT_ROOT / config_modelo['directorio']
        variantes_encontradas = []
        
        for archivo, param, nombre in config_modelo['variantes']:
            ruta_completa = directorio / archivo
            if ruta_completa.exists():
                variantes_encontradas.append((archivo, param, nombre, ruta_completa))
        
        if variantes_encontradas:
            modelos_encontrados[modelo_num] = {
                'nombre': config_modelo['nombre'],
                'variantes': variantes_encontradas
            }
    
    return modelos_encontrados


def evaluar_modelo(modelo_num, model_path, val_path, encoder_type=None, 
                   backbone=None, model_name=None, img_size=256, batch_size=32):
    """Evalúa un modelo usando validate_model.py."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "validate_model.py"),
        "--modelo", str(modelo_num),
        "--model_path", str(model_path),
        "--val_path", val_path,
        "--img_size", str(img_size),
        "--batch_size", str(batch_size),
        "--output_dir", str(PROJECT_ROOT / "outputs")
    ]
    
    # Agregar parámetros específicos según el modelo
    if modelo_num == 1 and encoder_type:
        cmd.extend(["--encoder_type", encoder_type])
    elif modelo_num in [2, 4, 5] and backbone:
        cmd.extend(["--backbone", backbone])
    elif modelo_num == 3 and model_name:
        cmd.extend(["--model_name", model_name])
    
    try:
        # Configurar encoding UTF-8 para evitar problemas en Windows
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Reemplazar caracteres problemáticos
            env=env,
            timeout=3600  # 1 hora máximo por modelo
        )
        
        if result.returncode == 0:
            # Intentar extraer métricas de la salida
            output = result.stdout
            return {
                'exito': True,
                'output': output,
                'error': None
            }
        else:
            # Extraer el mensaje de error más relevante
            error_msg = result.stderr
            if not error_msg or len(error_msg.strip()) == 0:
                error_msg = result.stdout
            
            # Buscar líneas de error relevantes
            error_lines = error_msg.split('\n')
            relevant_errors = []
            for line in error_lines:
                if any(keyword in line.lower() for keyword in ['error', 'exception', 'traceback', 'failed', 'cannot']):
                    relevant_errors.append(line.strip())
            
            if relevant_errors:
                error_msg = ' | '.join(relevant_errors[:3])  # Primeras 3 líneas relevantes
            
            return {
                'exito': False,
                'output': result.stdout,
                'error': error_msg[:500] if error_msg else 'Error desconocido'  # Limitar longitud
            }
    except subprocess.TimeoutExpired:
        return {
            'exito': False,
            'output': None,
            'error': 'Timeout: El modelo tardó más de 1 hora en evaluarse'
        }
    except Exception as e:
        return {
            'exito': False,
            'output': None,
            'error': str(e)
        }


def leer_resultados_json(output_dir):
    """Lee los resultados JSON más recientes del directorio de salida."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    # Buscar archivos JSON de validación
    json_files = list(output_path.glob("validation_results_*.json"))
    if not json_files:
        return None
    
    # Ordenar por fecha de modificación (más reciente primero)
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Leer el más reciente
    try:
        with open(json_files[0], 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error leyendo JSON: {e}")
        return None


def generar_resumen_completo(resultados, output_file):
    """Genera un resumen completo de todas las evaluaciones."""
    resumen = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_modelos_evaluados': len(resultados),
        'modelos': []
    }
    
    # Organizar resultados por modelo
    for modelo_num, variantes in resultados.items():
        modelo_info = {
            'modelo': modelo_num,
            'nombre': MODELOS_CONFIG[modelo_num]['nombre'],
            'variantes': []
        }
        
        for variante_info in variantes:
            variante_data = {
                'nombre': variante_info['nombre'],
                'archivo': variante_info['archivo'],
                'exito': variante_info['exito']
            }
            
            if variante_info['exito'] and variante_info.get('metricas'):
                variante_data['metricas'] = variante_info['metricas']
            
            if not variante_info['exito']:
                variante_data['error'] = variante_info.get('error', 'Error desconocido')
            
            modelo_info['variantes'].append(variante_data)
        
        resumen['modelos'].append(modelo_info)
    
    # Guardar resumen
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    
    return resumen


def imprimir_tabla_resumen(resultados):
    """Imprime una tabla resumen de todas las métricas."""
    print("\n" + "="*100)
    print("RESUMEN DE EVALUACIÓN - TODOS LOS MODELOS")
    print("="*100)
    
    # Encabezado de la tabla
    print(f"\n{'Modelo':<15} {'Variante':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Estado':<10}")
    print("-" * 100)
    
    # Ordenar por número de modelo
    for modelo_num in sorted(resultados.keys()):
        variantes = resultados[modelo_num]
        
        for variante in variantes:
            modelo_str = f"Modelo {modelo_num}"
            variante_str = variante['nombre']
            
            if variante['exito'] and variante.get('metricas'):
                metrics = variante['metricas'].get('metrics', {})
                accuracy = metrics.get('accuracy', 0) * 100
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1_score', 0)
                
                print(f"{modelo_str:<15} {variante_str:<25} {accuracy:>10.2f}% {precision:>11.4f} {recall:>11.4f} {f1:>11.4f} {'[OK]':<10}")
            else:
                error_msg = variante.get('error', 'Error')[:30]
                print(f"{modelo_str:<15} {variante_str:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'[FALLO]':<10}")
                if error_msg:
                    print(f"{'':<15} {'':<25} {'':<12} {'':<12} {'':<12} {'':<12} {error_msg}")
    
    print("="*100)


def generar_reporte_csv(resultados, output_file):
    """Genera un reporte CSV con todas las métricas."""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Encabezado
        writer.writerow([
            'Modelo', 'Nombre Modelo', 'Variante', 'Archivo',
            'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Loss',
            'TN', 'FP', 'FN', 'TP', 'Estado', 'Error'
        ])
        
        # Datos
        for modelo_num in sorted(resultados.keys()):
            nombre_modelo = MODELOS_CONFIG[modelo_num]['nombre']
            variantes = resultados[modelo_num]
            
            for variante in variantes:
                row = [
                    f"Modelo {modelo_num}",
                    nombre_modelo,
                    variante['nombre'],
                    variante['archivo']
                ]
                
                if variante['exito'] and variante.get('metricas'):
                    metrics = variante['metricas'].get('metrics', {})
                    cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
                    
                    row.extend([
                        f"{metrics.get('accuracy', 0):.4f}",
                        f"{metrics.get('precision', 0):.4f}",
                        f"{metrics.get('recall', 0):.4f}",
                        f"{metrics.get('f1_score', 0):.4f}",
                        f"{metrics.get('loss', 0):.6f}",
                        str(cm[0][0]),  # TN
                        str(cm[0][1]),  # FP
                        str(cm[1][0]),  # FN
                        str(cm[1][1]),  # TP
                        'OK',
                        ''
                    ])
                else:
                    row.extend([
                        'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                        'N/A', 'N/A', 'N/A', 'N/A',
                        'FALLO',
                        variante.get('error', 'Error desconocido')
                    ])
                
                writer.writerow(row)
    
    print(f"OK: Reporte CSV guardado en: {output_file}")


def generar_reporte_html(resultados, output_file):
    """Genera un reporte HTML con todas las métricas."""
    html = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Evaluación - Todos los Modelos</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .ok {
            color: green;
            font-weight: bold;
        }
        .fail {
            color: red;
            font-weight: bold;
        }
        .metric {
            text-align: right;
        }
        .summary {
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <h1>Reporte de Evaluación - Todos los Modelos</h1>
    <div class="summary">
        <p><strong>Fecha:</strong> """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <p><strong>Total de modelos evaluados:</strong> """ + str(sum(len(v) for v in resultados.values())) + """</p>
    </div>
    <table>
        <thead>
            <tr>
                <th>Modelo</th>
                <th>Variante</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Loss</th>
                <th>Estado</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # Generar filas de la tabla
    for modelo_num in sorted(resultados.keys()):
        nombre_modelo = MODELOS_CONFIG[modelo_num]['nombre']
        variantes = resultados[modelo_num]
        
        for variante in variantes:
            html += "            <tr>\n"
            html += f"                <td>Modelo {modelo_num}<br><small>{nombre_modelo}</small></td>\n"
            html += f"                <td>{variante['nombre']}</td>\n"
            
            if variante['exito'] and variante.get('metricas'):
                metrics = variante['metricas'].get('metrics', {})
                html += f"                <td class='metric'>{metrics.get('accuracy', 0)*100:.2f}%</td>\n"
                html += f"                <td class='metric'>{metrics.get('precision', 0):.4f}</td>\n"
                html += f"                <td class='metric'>{metrics.get('recall', 0):.4f}</td>\n"
                html += f"                <td class='metric'>{metrics.get('f1_score', 0):.4f}</td>\n"
                html += f"                <td class='metric'>{metrics.get('loss', 0):.6f}</td>\n"
                html += "                <td class='ok'>OK</td>\n"
            else:
                html += "                <td class='metric'>N/A</td>\n"
                html += "                <td class='metric'>N/A</td>\n"
                html += "                <td class='metric'>N/A</td>\n"
                html += "                <td class='metric'>N/A</td>\n"
                html += "                <td class='metric'>N/A</td>\n"
                html += f"                <td class='fail'>FALLO<br><small>{variante.get('error', 'Error')[:50]}</small></td>\n"
            
            html += "            </tr>\n"
    
    html += """        </tbody>
    </table>
</body>
</html>"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"OK: Reporte HTML guardado en: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluar todos los modelos entrenados con todas sus variantes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluar todos los modelos encontrados
  python evaluate_all_models.py --val_path E:\\Dataset\\Validacion_procesadas

  # Evaluar solo modelos específicos
  python evaluate_all_models.py --val_path E:\\Dataset\\Validacion_procesadas --modelos 1 2 3

  # Con parámetros personalizados
  python evaluate_all_models.py --val_path E:\\Dataset\\Validacion_procesadas --batch_size 64 --img_size 224
        """
    )
    
    parser.add_argument('--val_path', type=str, required=True,
                       help='Ruta al directorio de validación (debe contener carpetas normal/ y fallas/)')
    parser.add_argument('--modelos', type=int, nargs='+', default=None,
                       choices=[1, 2, 3, 4, 5],
                       help='Modelos específicos a evaluar (por defecto: todos los encontrados)')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Tamaño de los parches (default: 256, Modelo 3 usa 224 automáticamente)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamaño de batch (default: 32)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directorio para guardar resultados (default: outputs/)')
    
    args = parser.parse_args()
    
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
    
    # Encontrar modelos disponibles
    print("\n" + "="*100)
    print("BÚSQUEDA DE MODELOS ENTRENADOS")
    print("="*100)
    modelos_disponibles = encontrar_modelos_disponibles()
    
    if not modelos_disponibles:
        print("ERROR: No se encontraron modelos entrenados.")
        return
    
    # Filtrar por modelos solicitados
    if args.modelos:
        modelos_disponibles = {k: v for k, v in modelos_disponibles.items() if k in args.modelos}
    
    # Mostrar modelos encontrados
    total_variantes = sum(len(v['variantes']) for v in modelos_disponibles.values())
    print(f"\nModelos encontrados: {len(modelos_disponibles)}")
    print(f"Total de variantes: {total_variantes}")
    print("\nModelos a evaluar:")
    for modelo_num, info in modelos_disponibles.items():
        print(f"  - {info['nombre']}: {len(info['variantes'])} variantes")
        for archivo, param, nombre, ruta in info['variantes']:
            print(f"      • {nombre} ({archivo})")
    
    print("\n" + "="*100)
    print("INICIANDO EVALUACIÓN")
    print("="*100)
    
    # Directorio de salida
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluar cada modelo
    resultados = {}
    inicio_total = time.time()
    
    for modelo_num, info in sorted(modelos_disponibles.items()):
        print(f"\n{'='*100}")
        print(f"{info['nombre']}")
        print(f"{'='*100}")
        
        resultados[modelo_num] = []
        
        for idx, (archivo, param, nombre, ruta_completa) in enumerate(info['variantes'], 1):
            print(f"\n[{idx}/{len(info['variantes'])}] Evaluando: {nombre}")
            print(f"Archivo: {archivo}")
            
            inicio = time.time()
            
            # Determinar parámetros según el modelo
            encoder_type = None
            backbone = None
            model_name = None
            
            if modelo_num == 1:
                encoder_type = param
            elif modelo_num in [2, 4, 5]:
                backbone = param
            elif modelo_num == 3:
                model_name = param
            
            # ViT requiere imágenes de 224x224, otros modelos pueden usar 256
            img_size_actual = 224 if modelo_num == 3 else args.img_size
            
            # Evaluar modelo
            resultado = evaluar_modelo(
                modelo_num=modelo_num,
                model_path=str(ruta_completa),
                val_path=args.val_path,
                encoder_type=encoder_type,
                backbone=backbone,
                model_name=model_name,
                img_size=img_size_actual,
                batch_size=args.batch_size
            )
            
            tiempo = time.time() - inicio
            
            # Leer métricas del JSON generado
            metricas = None
            if resultado['exito']:
                # Esperar un momento para que se guarde el JSON
                time.sleep(1)
                metricas = leer_resultados_json(output_dir)
            
            # Guardar resultado
            resultados[modelo_num].append({
                'nombre': nombre,
                'archivo': archivo,
                'ruta': str(ruta_completa),
                'exito': resultado['exito'],
                'tiempo': tiempo,
                'metricas': metricas,
                'error': resultado.get('error')
            })
            
            if resultado['exito']:
                if metricas:
                    acc = metricas.get('metrics', {}).get('accuracy', 0) * 100
                    f1 = metricas.get('metrics', {}).get('f1_score', 0)
                    print(f"OK: Evaluación completada en {tiempo:.1f}s - Accuracy: {acc:.2f}%, F1: {f1:.4f}")
                else:
                    print(f"OK: Evaluación completada en {tiempo:.1f}s (métricas no disponibles)")
            else:
                error_msg = resultado.get('error', 'Error desconocido')
                # Limpiar el mensaje de error para mostrar solo lo relevante
                if 'UnicodeEncodeError' in error_msg:
                    error_msg = "Error de codificación (ya corregido)"
                elif 'RuntimeError' in error_msg or 'size mismatch' in error_msg.lower():
                    error_msg = f"Error de tamaño/modelo: {error_msg[:150]}"
                elif 'FileNotFoundError' in error_msg or 'No such file' in error_msg:
                    error_msg = f"Archivo no encontrado: {error_msg[:150]}"
                
                print(f"ERROR: {error_msg[:200]}")
    
    tiempo_total = time.time() - inicio_total
    
    # Generar resumen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resumen_file = output_dir / f"resumen_evaluacion_completa_{timestamp}.json"
    csv_file = output_dir / f"reporte_evaluacion_completa_{timestamp}.csv"
    html_file = output_dir / f"reporte_evaluacion_completa_{timestamp}.html"
    
    print("\n" + "="*100)
    print("GENERANDO REPORTES")
    print("="*100)
    
    resumen = generar_resumen_completo(resultados, resumen_file)
    generar_reporte_csv(resultados, csv_file)
    generar_reporte_html(resultados, html_file)
    
    # Imprimir tabla resumen
    imprimir_tabla_resumen(resultados)
    
    # Estadísticas finales
    total_evaluados = sum(len(v) for v in resultados.values())
    exitosos = sum(1 for v in resultados.values() for var in v if var['exito'])
    fallidos = total_evaluados - exitosos
    
    print(f"\n{'='*100}")
    print("ESTADÍSTICAS FINALES")
    print(f"{'='*100}")
    print(f"Total de modelos evaluados: {total_evaluados}")
    print(f"  OK - Exitosos: {exitosos}")
    print(f"  ERROR - Fallidos: {fallidos}")
    print(f"Tiempo total: {int(tiempo_total // 60)} min {tiempo_total % 60:.1f} seg")
    print(f"\nArchivos generados:")
    print(f"  - Resumen JSON: {resumen_file}")
    print(f"  - Reporte CSV: {csv_file}")
    print(f"  - Reporte HTML: {html_file}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()

