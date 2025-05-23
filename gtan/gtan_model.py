import os
import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from torch.optim.lr_scheduler import MultiStepLR

from .gtan_model import GraphAttnModel
from .gtan_lpa import load_lpa_subtensor
from . import early_stopper


def gtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features):
    device = args['device']
    graph = graph.to(device)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    oof_predictions  = torch.zeros(len(feat_df), 2, device=device)
    test_predictions = torch.zeros(len(feat_df), 2, device=device)

    kfold = StratifiedKFold(
        n_splits=args['n_fold'], shuffle=True, random_state=args['seed']
    )
    y_target = labels.iloc[train_idx].values

    # Prepara tensores de features
    num_feat = torch.from_numpy(feat_df.values).float().to(device)
    cat_feat = {
        col: torch.from_numpy(feat_df[col].values).long().to(device)
        for col in cat_features
    }
    labels = torch.from_numpy(labels.values).long().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    
    # Cross-validation
    for fold, (trn_idx, val_idx) in enumerate(
        kfold.split(feat_df.iloc[train_idx], y_target)
    ):
        print(f'=== Training fold {fold+1}/{args["n_fold"]} ===')
        epoch_logs = []
        trn_idx = torch.tensor(np.array(train_idx)[trn_idx], dtype=torch.long, device=device)
        val_idx = torch.tensor(np.array(train_idx)[val_idx], dtype=torch.long, device=device)

        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dl = NodeDataLoader(
            graph, trn_idx, train_sampler,
            batch_size=args['batch_size'], shuffle=True, device=device
        )
        val_dl = NodeDataLoader(
            graph, val_idx, train_sampler,
            batch_size=args['batch_size'], shuffle=False, device=device
        )

        drop_cfg = args['dropout']
        if isinstance(drop_cfg, (list, tuple)):
            drop_tuple = tuple(drop_cfg)
        else:
            drop_tuple = (drop_cfg, drop_cfg)

        # Modelo y optimizador
        model = GraphAttnModel(
            in_feats=feat_df.shape[1],
            hidden_dim=args['hid_dim'] // 4,
            n_layers=args['n_layers'],
            n_classes=2,
            heads=[4] * args['n_layers'],
            activation=nn.PReLU(),
            drop=drop_tuple,
            gated=args['gated'],
            ref_df=feat_df,
            cat_features=cat_feat,
            device=device
        ).to(device)

        lr = args['lr'] * np.sqrt(args['batch_size'] / 1024)
        optimizer = optim.Adam(model.parameters(),lr=float(lr),weight_decay=float(args['wd']))
        scheduler = MultiStepLR(optimizer, milestones=[4000, 12000], gamma=0.3)

        stopper = early_stopper(patience=args['early_stopping'], verbose=True)

        # Bucles de entrenamiento
        for epoch in range(args['max_epochs']):
            model.train()
            train_losses = []
            for step, (input_nodes, seeds, blocks) in enumerate(train_dl):
                batch_in, batch_work, batch_lbls, lpa_lbls = load_lpa_subtensor(
                    num_feat, cat_feat, labels, seeds, input_nodes, device
                )
                blocks = [b.to(device) for b in blocks]

                logits = model(blocks, batch_in, lpa_lbls, batch_work)
                mask = batch_lbls == 2
                logits = logits[~mask]
                targets = batch_lbls[~mask]

                loss = loss_fn(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                if step % 10 == 0:
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == targets).float().mean()
                    scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                    try:
                        ap  = average_precision_score(targets.cpu().numpy(), scores)
                        auc = roc_auc_score(targets.cpu().numpy(), scores)
                        print(f"[Epoch {epoch:03d} Batch {step:04d}] "
                              f"loss={np.mean(train_losses):.4f} ap={ap:.4f} "
                              f"acc={acc:.4f} auc={auc:.4f}")
                    except:
                        pass

            # Validación
            model.eval()
            val_loss, val_acc_sum, val_count = 0.0, 0.0, 0
            with torch.no_grad():
                for _, (inp, seeds, blks) in enumerate(val_dl):
                    bi, bw, bl, ll = load_lpa_subtensor(
                        num_feat, cat_feat, labels, seeds, inp, device
                    )
                    blks = [b.to(device) for b in blks]
                    vl = model(blks, bi, ll, bw)
                    oof_predictions[seeds] = vl

                    mask = bl == 2
                    vl = vl[~mask]
                    tgt = bl[~mask]
                    val_loss += loss_fn(vl, tgt).item() * tgt.shape[0]
                    val_acc_sum += (torch.argmax(vl, 1) == tgt).sum().item()
                    val_count += tgt.shape[0]

            mean_val_loss = val_loss / val_count
            stopper.earlystop(mean_val_loss, model)
            print(f"-- Validation loss: {mean_val_loss:.4f}")
            epoch_logs.append({"epoch": epoch,"train_loss": float(np.mean(train_losses)),"val_loss":   float(mean_val_loss)})
            if stopper.is_earlystop:
                print(">>> Early stopping")
                break

        df_loss = pd.DataFrame(epoch_logs)
        df_loss.to_csv(results_dir / f"fold{fold+1}_losses.csv", index=False)
        print(f">>> Fold {fold+1}: losses guardados en results/fold{fold+1}_losses.csv")
        best_model = stopper.best_model.to(device).eval()
        test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        test_dl = NodeDataLoader(
            graph,
            torch.tensor(test_idx, dtype=torch.long, device=device),
            test_sampler,
            batch_size=args['batch_size'],
            shuffle=False,
            device=device
        )
        with torch.no_grad():
            for _, (inp, seeds, blks) in enumerate(test_dl):
                bi, bw, bl, ll = load_lpa_subtensor(
                    num_feat, cat_feat, labels, seeds, inp, device
                )
                blks = [b.to(device) for b in blks]
                tv = best_model(blks, bi, ll, bw)
                test_predictions[seeds] += tv
        
        y_all   = labels.cpu().numpy()[test_idx]
        scores  = torch.softmax(test_predictions, dim=1)[test_idx, 1] \
                        .detach().cpu().numpy()

        mask = (y_all != 2)
        df_pred = pd.DataFrame({
            "fold":    fold + 1,
            "y_true":  y_all[mask],
            "y_score": scores[mask]
        })
        df_pred.to_csv(results_dir / f"fold{fold+1}_preds.csv", index=False)
        print(f">>> Fold {fold+1}: preds guardados en results/fold{fold+1}_preds.csv")
        

    y_all_final = labels.cpu().numpy()[test_idx]
    scores_final = torch.softmax(test_predictions, dim=1)[test_idx, 1].detach().cpu().numpy()

    mask_final = (y_all_final != 2)
    y_true = y_all_final[mask_final]
    scores = scores_final[mask_final]
    preds  = (scores > 0.5).astype(int)  

    print("Final test AUC :", roc_auc_score(y_true, scores))
    print("Final test F1  :", f1_score(y_true, preds, average="macro"))
    print("Final test AP  :", average_precision_score(y_true, scores))

    if args.get("save_model", False):
        out_path = args.get("model_path", "model.pth")
        torch.save(best_model.state_dict(), out_path)
        print(f">>> Modelo final guardado en '{out_path}'")




def load_gtan_data(dataset: str, test_size: float):
    if dataset != "S-FFSD":
        raise ValueError(f"Dataset '{dataset}' no soportado. Sólo 'S-FFSD'.")

    prefix = os.path.join(os.path.dirname(__file__), "..", "data/")
    df = pd.read_csv(prefix + "S-FFSDneofull.csv")
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    data = df[df["Labels"] <= 2].reset_index(drop=True)

    # Construcción del grafo
    src_all, tgt_all = [], []
    for col in ["Source", "Target", "Location", "Type"]:
        for _, grp in data.groupby(col):
            idxs = grp.sort_values("Time").index.to_numpy()
            for i in range(len(idxs)):
                for j in range(1, 4):
                    if i + j < len(idxs):
                        src_all.append(idxs[i])
                        tgt_all.append(idxs[i + j])
    graph = dgl.graph((src_all, tgt_all))
    graph = dgl.add_self_loop(graph)

    # Codificación de categorías
    cat_features = ["Target", "Location", "Type"]
    for col in cat_features + ["Source"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    feat_data = data.drop("Labels", axis=1)
    labels    = data["Labels"]
    indices   = list(range(len(labels)))

    graph.ndata['feat']  = torch.from_numpy(feat_data.values).float()
    graph.ndata['label'] = torch.from_numpy(labels.values).long()

    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=2
    )

    # Guarda el grafo por si quieres recargarlo
    graph_path = prefix + f"graph-{dataset}.bin"
    dgl.data.utils.save_graphs(graph_path, [graph])

    return feat_data, labels, train_idx, test_idx, graph, cat_features
