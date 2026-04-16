import hashlib
import json
import os
import time
import torch

from typing import Any

_PREPROCESS_CACHE_VERSION = 1
_PREPROCESS_CACHE_KEY_FIELDS = (
    'dataset',
    'ppr_backend',
    'ppr_alpha',
    'ppr_topk',
    'ppr_num_iterations',
    'ppr_batch_size',
    'ppr_iter_topk',
    'ppr_eps',
    'n_parts',
    'related_nodes_topk_rate',
    'attn_type',
    'struct_enc',
    'max_dist',
)


def _preprocess_cache_dir(args):
    cache_dir = os.path.join(args.dataset_dir, args.dataset, 'preprocess_cache')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _args_snapshot(args, world_size: int):
    snapshot = {key: getattr(args, key) for key in _PREPROCESS_CACHE_KEY_FIELDS}
    snapshot['world_size'] = int(world_size)
    return snapshot


def compute_preprocess_cache_key(args, world_size: int):
    snapshot = _args_snapshot(args, world_size)
    key_json = json.dumps(snapshot, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(key_json.encode('utf-8')).hexdigest(), snapshot


def preprocess_cache_path(args, cache_key: str):
    return os.path.join(_preprocess_cache_dir(args), f'preprocess_{cache_key}.pt')


def build_preprocess_cache_payload(struct_info: Any, args_snapshot, cache_key: str):
    return {
        'cache_version': _PREPROCESS_CACHE_VERSION,
        'cache_key': cache_key,
        'args_snapshot': args_snapshot,
        'num_nodes': struct_info.num_nodes,
        'graph_in_degree': struct_info.graph_in_degree,
        'graph_out_degree': struct_info.graph_out_degree,
        'sorted_ppr_matrix': struct_info.sorted_ppr_matrix,
        'graph_edge_index': struct_info.graph_edge_index,
        'graph_csr_data': struct_info.graph_csr_data,
        'wm': {
            'partitioned_results': struct_info.wm.partitioned_results,
            'sub_edge_index_for_partition_results': struct_info.wm.sub_edge_index_for_partition_results,
            'dup_nodes_per_partition': struct_info.wm.dup_nodes_per_partition,
        },
    }


def save_preprocess_cache(args, struct_info: Any, cache_key: str, args_snapshot):
    cache_path = preprocess_cache_path(args, cache_key)
    payload = build_preprocess_cache_payload(struct_info, args_snapshot, cache_key)
    save_start = time.time()
    torch.save(payload, cache_path)
    return cache_path, time.time() - save_start


def load_preprocess_cache(args, graph_in_degree, graph_out_degree, edge_index=None, edge_csr_data=None, num_nodes=None, world_size: int = 1):
    cache_key, args_snapshot = compute_preprocess_cache_key(args, world_size)
    cache_path = preprocess_cache_path(args, cache_key)
    if not os.path.exists(cache_path):
        return None, cache_key, cache_path, args_snapshot, 0.0

    load_start = time.time()
    payload = torch.load(cache_path, map_location='cpu')
    load_time = time.time() - load_start
    if payload.get('cache_version') != _PREPROCESS_CACHE_VERSION:
        return None, cache_key, cache_path, args_snapshot, load_time
    if payload.get('cache_key') != cache_key:
        return None, cache_key, cache_path, args_snapshot, load_time

    cached_snapshot = payload.get('args_snapshot', {})
    if cached_snapshot != args_snapshot:
        return None, cache_key, cache_path, args_snapshot, load_time

    return payload, cache_key, cache_path, args_snapshot, load_time
