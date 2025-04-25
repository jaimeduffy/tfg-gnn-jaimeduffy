
# Graph Neural Networks: Deep learning para entornos no estructurados.

Proyecto de TFG de Jaime Duffy Panés | jaidufpan@alum.us.es


Este repositorio contiene la implementación de los experimentos realizados como parte de mi trabajo de fin de grado sobre Graph Neural Networks (GNNs).

---

## Estructura del repositorio

```
TFG-GTAN-JAIMEDUFFY/
├── config/
│   └── gtan_cfg.yaml          # Configuración del modelo y parámetros
├── data/
│   └── S-FFSD.zip             # Dataset original comprimido
├── feature_engineering/
│   └── data_process.py        # Preprocesamiento del dataset
├── gtan/
│   ├── __init__.py
│   ├── gtan_model.py          # Definición de la arquitectura GTAN
│   └── gtan_main.py           # Entrenamiento, validación y test
├── notebooks/
│   └── gnn-examples.ipynb     # Implementaciones con GCN, GAT y predicción de enlaces
├── main.py                    # Punto de entrada para lanzar el experimento
├── environment.txt           # Dependencias necesarias
└── README.md
```

---

## ¿Cómo ejecutar el proyecto?

### 1. Clona el repositorio

```bash
git clone https://github.com/jaimeduffy/tfg-gnn-jaimeduffy.git
cd tfg-gnn-jaimeduffy
```

### 2. Crea y activa un entorno virtual (recomendado con conda)

```bash
conda create -n antifraud-env python=3.7
conda activate antifraud-env
pip install -r environment.txt
```

### 3. Descomprime el dataset

```bash
unzip data/S-FFSD.zip -d data/
```

### 4. Ejecuta el preprocesamiento

```bash
python feature_engineering/data_process.py
```

### 5. Entrena el modelo GTAN

```bash
python main.py
```

---

## Resultados esperados

El modelo imprimirá durante el entrenamiento los valores de pérdida, AUC, F1 y AP en validación y test. También puedes activar el guardado del modelo entrenado en `config/gtan_cfg.yaml` con:

```yaml
save_model: true
model_path: gtan_trained.pth
```

---

## Créditos

Este trabajo se ha basado en el código original publicado por [AI4Risk](https://github.com/AI4Risk/antifraud), adaptado, depurado y documentado para su uso académico.

El desarrollo forma parte del **Trabajo de Fin de Grado** de Jaime Duffy (Universidad de Sevilla, 2025).

