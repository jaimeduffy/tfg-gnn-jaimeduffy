import os
import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
import dgl
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from gtan import early_stopper
from gtan.gtan_model import GraphAttnModel

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_gtan_data(config):
    dataset = config["dataset"]
    assert dataset == "S-FFSD", "Este script solo admite S-FFSD."

    root_path = os.path.join(os.path.dirname(__file__), "..", "data")
    graph_path = os.path.join(root_path, "graph-S-FFSD.bin")
    neigh_feat_path = os.path.join(root_path, "S-FFSD_neigh_feat.csv")

    g, _ = dgl.load_graphs(graph_path)
    g = g[0]
    feat_data = torch.tensor(pd.read_csv(neigh_feat_path).values).float()
    labels = g.ndata['label']

    # Categóricas (no usadas en GTAN puro, pero se mantienen por compatibilidad)
    cat_data = torch.zeros_like(feat_data)

    return g, feat_data, cat_data, labels

def load_subtensor(g, feat_data, cat_data, labels, indices, device):
    return (
        g,
        feat_data[indices].to(device),
        cat_data[indices].to(device),
        labels[indices].to(device)
    )

def train_and_evaluate(config_path="config/gtan_cfg.yaml"):
    args = load_config(config_path)
    set_seed(args["seed"])

    if args["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args["device"])
    print(f"Usando dispositivo: {device}")

    g, feat_data, cat_data, labels = load_gtan_data(args)
    skf = StratifiedKFold(n_splits=args["n_fold"], shuffle=True, random_state=args["seed"])

    auc_list, f1_list, ap_list = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(feat_data, labels)):
        print(f"\n===== Fold {fold + 1}/{args['n_fold']} =====")

        train_idx = torch.tensor(train_idx)
        test_idx = torch.tensor(test_idx)

        model = GraphAttnModel(
            in_feats=feat_data.shape[1],
            hidden_dim=args['hidden_dim'],
            n_classes=args['n_classes'],
            heads=args['heads'],
            n_layers=args['n_layers'],
            activation=nn.ReLU(),
            drop=args['dropout'],
            device=device,
            gated=args['gated'],
            ref_df=feat_data,
            cat_features=cat_data
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=args["wd"])
        stopper = early_stopper(patience=args["early_stopping"])

        epoch_metrics = []
        for epoch in range(args["max_epochs"]):
            model.train()
            optimizer.zero_grad()
            
            # Entrenamiento
            g_batch, feat_batch, cat_batch, label_batch = load_subtensor(
                g, feat_data, cat_data, labels, train_idx, device
            )
            logits = model(g_batch, feat_batch, cat_batch)
            train_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_batch.float())
            train_loss.backward()
            optimizer.step()
            
            # Cálculo del AUC en entrenamiento
            train_probs = torch.sigmoid(logits)
            train_auc = roc_auc_score(label_batch.cpu(), train_probs.cpu())
            
            # Validación
            model.eval()
            with torch.no_grad():
                _, val_feat, val_cat, val_label = load_subtensor(
                    g, feat_data, cat_data, labels, test_idx, device
                )
                val_logits = model(g, val_feat, val_cat)
                val_probs = torch.sigmoid(val_logits)
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(val_logits, val_label.float()).item()
                val_auc = roc_auc_score(val_label.cpu(), val_probs.cpu())

            print(f"Epoch {epoch+1} | Train Loss: {train_loss.item():.4f} | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

            epoch_metrics.append({
                "epoch": epoch + 1,
                "train_loss": train_loss.item(),
                "train_auc": train_auc,
                "val_loss": val_loss,
                "val_auc": val_auc
            })

            stopper(val_auc, model)
            if stopper.early_stop:
                print("Early stopping.")
                break

        # Cargar mejor modelo y evaluar
        model.load_state_dict(torch.load("early_stop_model.pth"))
        model.eval()
        with torch.no_grad():
            _, test_feat, test_cat, test_label = load_subtensor(
                g, feat_data, cat_data, labels, test_idx, device
            )
            test_logits = model(g, test_feat, test_cat)
            test_probs = torch.sigmoid(test_logits)

        auc = roc_auc_score(test_label.cpu(), test_probs.cpu())
        f1 = f1_score(test_label.cpu(), test_probs.cpu() > 0.5)
        ap = average_precision_score(test_label.cpu(), test_probs.cpu())

        print(f"Test AUC: {auc:.4f} | F1: {f1:.4f} | AP: {ap:.4f}")

        auc_list.append(auc)
        f1_list.append(f1)
        ap_list.append(ap)

        with open(f"fold_{fold+1}_metrics.json", "w") as f:
            json.dump(epoch_metrics, f, indent=4)

    # Resultados finales
    print("\n===== Resultados Promedio =====")
    print(f"AUC: {np.mean(auc_list):.4f}")
    print(f"F1 : {np.mean(f1_list):.4f}")
    print(f"AP : {np.mean(ap_list):.4f}")

    # Guardar el modelo final si está activado
    if args.get("save_model", False):
        torch.save(model.state_dict(), args.get("model_path", "gtan_trained.pth"))
        print(f"Modelo guardado en {args.get('model_path', 'gtan_trained.pth')}")

if __name__ == "__main__":
    train_and_evaluate()
