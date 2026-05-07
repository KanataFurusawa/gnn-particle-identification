import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

def create_graph_for_event(i, t, ch_x, ch_y, trkp, eHit_x, eHit_z, label):
    num_hits = len(t[i])
    if num_hits == 0:
        return None

    node_features = []
    for j in range(num_hits):
        node_features.append([
            t[i][j],
            ch_x[i][j] / 64.0,
            ch_y[i][j] / 8.0,
            float(trkp[i]),
            float(eHit_x[i]),
            float(eHit_z[i])
        ])

    x = torch.tensor(node_features, dtype=torch.float)

    # 完全グラフを作成
    edges = []

    for a in range(num_hits):
        for b in range(num_hits):

            if a != b:
                edges.append([a, b])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    y = torch.tensor([label], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)
