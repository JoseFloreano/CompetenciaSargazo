# Competencia Sargazo - Clasificacion de Sargazo en el Caribe Mexicano

Proyecto de clasificacion multiclase ordinal para estimar la cantidad de sargazo
en playas del Caribe mexicano a partir de imagenes extraidas de redes sociales.

**Responsable del dataset:** [Juan Irving Vasquez](https://jivg.org/)

## **Responsables del proyecto:**

**Apoyo institucional:** Consejo Nacional de Ciencia y Tecnologia

---

## Descripcion del Problema

Dado un conjunto de imagenes de playas, el modelo debe predecir el nivel de sargazo
presente segun la siguiente escala ordinal:

| Etiqueta | Clase     |
| -------- | --------- |
| 0        | Nada      |
| 1        | Bajo      |
| 2        | Moderado  |
| 3        | Abundante |
| 4        | Excesivo  |

La metrica principal de evaluacion es el **Quadratic Weighted Kappa (QWK)**, adecuada
para clasificacion ordinal ya que penaliza mas los errores entre clases distantes.

---

## Estructura del Repositorio

```
CompetenciaSargazo/
├── images/                     # Todas las imagenes (train + test)
├── labels/
│   ├── labels.csv              # Imagenes de entrenamiento con etiquetas (columnas: image_path, nivel)
│   └── test.csv                # Imagenes objetivo sin etiqueta (una columna, sin encabezado)
├── sargazo_classification.ipynb  # Notebook principal de entrenamiento e inferencia
├── requirements.txt            # Dependencias del proyecto
└── docs/
    ├── prompt_dis_modelo.md    # Especificaciones tecnicas del modelo
    └── Readme_ds_competencia.md
```

Archivos generados al ejecutar el notebook:

| Archivo                | Descripcion                                    |
| ---------------------- | ---------------------------------------------- |
| `best_model.pth`       | Pesos del mejor modelo segun QWK de validacion |
| `submission.csv`       | Predicciones del conjunto de test (entrega)    |
| `training_history.png` | Graficas de loss, F1 y QWK por epoca           |

---

## Metodologia

### Datos

- 2,717 imagenes de entrenamiento con clases desbalanceadas
- Division estratificada 80/20 para Train/Validacion
- Pesos de clase calculados por frecuencia inversa (`compute_class_weight`)

### Modelo

- **Arquitectura:** ResNet-50 con pesos preentrenados `IMAGENET1K_V2`
- **Transfer Learning en dos fases:**
  - Fase 1 (10 epocas): backbone congelado, solo se entrena la capa de clasificacion
  - Fase 2 (hasta 40 epocas): backbone descongelado con LR diferenciado (backbone `1e-5`, cabeza `1e-4`)

### Entrenamiento

- **Loss:** CrossEntropyLoss con pesos de clase
- **Optimizador:** AdamW
- **Scheduler:** ReduceLROnPlateau (monitorea QWK de validacion)
- **Precision mixta:** `torch.cuda.amp` (autocast + GradScaler) para VRAM 16 GB
- **Early stopping:** paciencia de 7 epocas sobre QWK de validacion

### Augmentacion

| Split      | Transformaciones                                                           |
| ---------- | -------------------------------------------------------------------------- |
| Train      | Resize 224x224, RandomHorizontalFlip, ColorJitter, RandomAffine, Normalize |
| Val / Test | Resize 224x224, Normalize (media y std ImageNet)                           |

---

## Requisitos

**Hardware:** NVIDIA RTX 5080 Laptop (16 GB VRAM) o superior con CUDA 12.4+

**Instalacion de dependencias:**

```bash
# 1. Instalar PyTorch con soporte CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Instalar el resto de dependencias
pip install -r requirements.txt
```

---

## Uso

1. Colocar las imagenes en `images/` y los archivos CSV en `labels/`.
2. Abrir `sargazo_classification.ipynb` en JupyterLab.
3. Ejecutar todas las celdas en orden.
4. El archivo `submission.csv` se genera automaticamente al final del notebook.

---

_Contributors: ver historial de commits_
