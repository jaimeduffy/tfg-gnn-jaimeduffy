
# TFG-GNN-JAIMEDUFFY

Este repositorio contiene la implementación del modelo **GTAN (Graph Temporal Attention Network)** aplicado al conjunto de datos **S-FFSD**, como parte de mi trabajo de fin de grado sobre Graph Neural Networks (GNNs).

---

## ¿Qué es GTAN?

**GTAN** es un modelo graph neural networks que incorpora mecanismos de atención temporal para detectar comportamientos fraudulentos en escenarios con datos multirrelacionales y dependencias temporales. El modelo se basa en el paper:

> **GTAN: A Graph Temporal Attentive Network for Fraud Detection**  
> https://arxiv.org/abs/2412.18287  
> [Código original del paper](https://github.com/AI4Risk/antifraud)

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
├── main.py                    # Punto de entrada para lanzar el experimento
├── requirements.txt           # Dependencias necesarias
└── README.md
```

---

## ¿Cómo ejecutar el proyecto?

### 1. Clona el repositorio

```bash
git clone https://github.com/jaimeduffy/tfg-gtan-jaimeduffy.git
cd tfg-gtan-jaimeduffy
```

### 2. Crea y activa un entorno virtual (recomendado con conda)

```bash
conda create -n antifraud-env python=3.7
conda activate antifraud-env
pip install -r requirements.txt
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

