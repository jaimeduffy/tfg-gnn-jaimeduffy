
# Graph Neural Networks: Deep learning para entornos no estructurados.

### Proyecto de TFG de Jaime Duffy Panés | jaidufpan@alum.us.es


Este repositorio contiene la implementación de los experimentos realizados a lo largo de mi trabajo de fin de grado sobre Graph Neural Networks (GNNs).

El repositorio se divide en dos partes:

- Un notebook de jupyter **gnn-examples.ipynb** en la carpeta **/notebooks** con las implementaciones de modelos básicos como GCN y GAT sobre el dataset Cora. Estas implementaciones están pensadas como introducción conceptual a las GNNs y se pueden ejecutar de forma independiente en Google Colab o en cualquier entorno local con Jupyter Notebook sin necesidad de instalación adicional.
  
- La implementación principal que consiste en la aplicación del modelo **GTAN (Graph Temporal Attention Network)** para la detección de fraude financiero sobre el conjunto de datos **S-FFSD (Simulated Financial Fraud Semi-supervised Dataset)**. Esta parte requiere un entorno de desarrollo definido en el archivo `environment.txt`, y se recomienda ejecutarla localmente en un entorno virtual (por ejemplo, `conda`).

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
├── results/                   # Volcado de resultados
├── main.py                    # Punto de entrada para lanzar el experimento
├── environment.txt           # Dependencias necesarias
├── plot_results.py           # Generar gráficas
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
unzip /data/S-FFSD.zip
```

### 4. Ejecuta el preprocesamiento

```bash
python feature_engineering/data_process.py
```

### 5. Entrena el modelo GTAN

```bash
python main.py
```

### 6. Genera las gráficas finales

```bash
 python plot_results.py
```

---

## Créditos

Este trabajo se ha basado en el código original publicado por [AI4Risk](https://github.com/AI4Risk/antifraud), adaptado, depurado y documentado para su uso académico.

El desarrollo forma parte del **Trabajo de Fin de Grado** de Jaime Duffy (Universidad de Sevilla, 2025).

