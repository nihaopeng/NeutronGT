#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from core.pprPartition import export_graph_for_fora


def main():
    parser = argparse.ArgumentParser(description='Export a dataset graph into FORA input format.')
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./third_party/fora/data')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_dir) / args.dataset
    edge_index = torch.load(dataset_root / 'edge_index.pt', map_location='cpu')
    x_path = dataset_root / 'x.pt'
    y_path = dataset_root / 'y.pt'
    if x_path.exists():
        num_nodes = int(torch.load(x_path, map_location='cpu').shape[0])
    elif y_path.exists():
        num_nodes = int(torch.load(y_path, map_location='cpu').shape[0])
    else:
        raise FileNotFoundError(f'Cannot infer num_nodes because neither {x_path} nor {y_path} exists')

    graph_dir = Path(args.output_dir) / args.dataset
    graph_path, attr_path, ssquery_path = export_graph_for_fora(edge_index=edge_index, num_nodes=num_nodes, output_dir=graph_dir)
    print(f'graph.txt: {graph_path}')
    print(f'attribute.txt: {attr_path}')
    print(f'ssquery.txt: {ssquery_path}')


if __name__ == '__main__':
    main()
