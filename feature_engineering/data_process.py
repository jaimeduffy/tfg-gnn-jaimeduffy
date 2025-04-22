import pandas as pd
import numpy as np
import torch
import dgl
import os
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict


DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data/")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def featmap_gen(tmp_df):
    """Feature engineering sobre el dataset S-FFSD"""
    time_span = [2, 3, 5, 15, 20, 50, 100, 150,
                 200, 300, 864, 2590, 5100, 10000, 24000]
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []

    for trans_idx, trans_feat in tqdm(tmp_df.iterrows(), total=len(tmp_df)):
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.Amount
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            new_df[f'trans_at_avg_{tname}'] = correct_data['Amount'].mean()
            new_df[f'trans_at_totl_{tname}'] = correct_data['Amount'].sum()
            new_df[f'trans_at_std_{tname}'] = correct_data['Amount'].std()
            new_df[f'trans_at_bias_{tname}'] = temp_amt - correct_data['Amount'].mean()
            new_df[f'trans_at_num_{tname}'] = len(correct_data)
            new_df[f'trans_target_num_{tname}'] = len(correct_data.Target.unique())
            new_df[f'trans_location_num_{tname}'] = len(correct_data.Location.unique())
            new_df[f'trans_type_num_{tname}'] = len(correct_data.Type.unique())
        post_fe.append(new_df)

    return pd.DataFrame(post_fe)


def count_risk_neighs(graph: dgl.DGLGraph, risk_label: int = 1):
    """Cuenta número de vecinos con etiqueta de fraude"""
    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata['label'][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)
    return torch.Tensor(ret)


def feat_map(graph, edge_feat):
    """Extrae características basadas en vecinos"""
    tensor_list = []
    for idx in tqdm(range(graph.num_nodes())):
        neighs_1 = graph.predecessors(idx)
        neighs_2 = dgl.khop_in_subgraph(graph, idx, 2)[0].ndata[dgl.NID]
        neighs_2 = neighs_2[neighs_2 != idx]
        tensor = torch.FloatTensor([
            edge_feat[neighs_1, 0].sum().item(),
            edge_feat[neighs_2, 0].sum().item(),
            edge_feat[neighs_1, 1].sum().item(),
            edge_feat[neighs_2, 1].sum().item(),
        ])
        tensor_list.append(tensor)
    return torch.stack(tensor_list), ["1hop_degree", "2hop_degree", "1hop_riskstat", "2hop_riskstat"]


if __name__ == "__main__":
    set_seed(42)
    print("Procesando S-FFSD...")

    # Leer el dataset original
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSD.csv'))
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)
    data.to_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'), index=None)
    data = pd.read_csv(os.path.join(DATADIR, 'S-FFSDneofull.csv'))

    # Construcción del grafo
    out, alls, allt = [], [], []
    for column in ["Source", "Target", "Location", "Type"]:
        src, tgt = [], []
        for c_id, c_df in tqdm(data.groupby(column), desc=column):
            c_df = c_df.sort_values(by="Time")
            df_len = len(c_df)
            sorted_idxs = c_df.index
            src.extend([sorted_idxs[i] for i in range(df_len)
                        for j in range(3) if i + j < df_len])
            tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                        for j in range(3) if i + j < df_len])
        alls.extend(src)
        allt.extend(tgt)

    g = dgl.graph((np.array(alls), np.array(allt)))

    # Encoding y normalización
    for col in ["Source", "Target", "Location", "Type"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    labels = data["Labels"]
    feat_data = data.drop("Labels", axis=1)

    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

    # Guardar grafo binario
    dgl.data.utils.save_graphs(os.path.join(DATADIR, "graph-S-FFSD.bin"), [g])

    # Calcular features de vecinos
    print("Generando características de riesgo vecinal...")
    degree_feat = g.in_degrees().unsqueeze(1).float()
    risk_feat = count_risk_neighs(g).unsqueeze(1).float()
    edge_feat = torch.cat([degree_feat, risk_feat], dim=1)

    neigh_feat, feat_names = feat_map(g, edge_feat)
    full_feat = torch.cat((edge_feat, neigh_feat), dim=1).numpy()
    feat_names = ['degree', 'riskstat'] + feat_names

    full_feat[np.isnan(full_feat)] = 0
    df_feat = pd.DataFrame(full_feat, columns=feat_names)
    df_feat = pd.DataFrame(StandardScaler().fit_transform(df_feat), columns=feat_names)
    df_feat.to_csv(os.path.join(DATADIR, "S-FFSD_neigh_feat.csv"), index=False)

    print("Preprocesamiento completado.")
