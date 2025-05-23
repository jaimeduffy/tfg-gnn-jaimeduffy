#!/usr/bin/env python3

import yaml
from gtan.gtan_main import gtan_main, load_gtan_data

def main(config_path: str = "config/gtan_cfg.yaml"):
    # Carga de configuración
    with open(config_path, 'r') as f:
        args = yaml.safe_load(f)

    # Resolver device 'auto'
    import torch
    dev = args.get("device", "auto")
    if dev == "auto":
        args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        args["device"] = dev

    # Carga de datos para el dataset S-FFSD
    dataset   = args.get("dataset", "S-FFSD")
    test_size = args.get("test_size", 0.2)
    feat_df, labels, train_idx, test_idx, graph, cat_features = \
        load_gtan_data(dataset, test_size)

    # Entrenamiento y evaluación
    gtan_main(
        feat_df,
        graph,
        train_idx,
        test_idx,
        labels,
        args,
        cat_features
    )

if __name__ == "__main__":
    main()
