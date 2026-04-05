from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


_CHECKPOINT_COMPAT_KEYS = (
    'dataset',
    'model',
    'n_layers',
    'hidden_dim',
    'num_heads',
    'attn_type',
    'n_parts',
    'related_nodes_topk_rate',
    'ppr_topk',
)


@dataclass
class ResumeState:
    start_epoch: int = 0
    loss_mean_list: list[float] | None = None
    best_val: float = float('-inf')
    best_test: float = float('-inf')
    checkpoint_path: str | None = None


def checkpoint_dir(args) -> Path:
    base_dir = args.checkpoint_dir if args.checkpoint_dir else args.model_dir
    return Path(base_dir)


def checkpoint_paths(args, epoch: int | None = None) -> dict[str, Path]:
    ckpt_dir = checkpoint_dir(args)
    paths = {
        'dir': ckpt_dir,
        'last': ckpt_dir / 'last.pt',
        'best': ckpt_dir / 'best.pt',
    }
    if epoch is not None:
        paths['epoch'] = ckpt_dir / f'epoch_{epoch + 1}.pt'
    return paths


def resolve_resume_path(args) -> str | None:
    if args.resume_checkpoint:
        return args.resume_checkpoint
    if not args.resume_latest:
        return None

    paths = checkpoint_paths(args)
    if paths['last'].exists():
        return str(paths['last'])

    epoch_candidates = sorted(paths['dir'].glob('epoch_*.pt'))
    if epoch_candidates:
        return str(epoch_candidates[-1])
    raise FileNotFoundError(f'No checkpoint found under {paths["dir"]}')


def validate_resume_supported(args) -> None:
    if args.use_cache != 1:
        raise ValueError('Checkpoint resume v1 only supports fixed-window training with --use_cache 1.')


def ensure_checkpoint_dir(args) -> Path:
    ckpt_dir = checkpoint_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _validate_checkpoint_compatibility(args, args_snapshot: dict[str, Any]) -> None:
    if not args_snapshot:
        return
    mismatches = []
    for key in _CHECKPOINT_COMPAT_KEYS:
        current_value = getattr(args, key, None)
        saved_value = args_snapshot.get(key)
        if saved_value is None:
            continue
        if current_value != saved_value:
            mismatches.append(f'{key}: current={current_value}, checkpoint={saved_value}')
    if mismatches:
        mismatch_text = '; '.join(mismatches)
        raise ValueError(f'Checkpoint is incompatible with current run arguments: {mismatch_text}')


def load_training_checkpoint(args, model, optimizer, lr_scheduler, device) -> ResumeState:
    resume_path = resolve_resume_path(args)
    if resume_path is None:
        return ResumeState(loss_mean_list=[])

    validate_resume_supported(args)
    checkpoint = torch.load(resume_path, map_location=device)
    args_snapshot = checkpoint.get('args_snapshot', {})
    _validate_checkpoint_compatibility(args, args_snapshot)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    return ResumeState(
        start_epoch=int(checkpoint.get('epoch', -1)) + 1,
        loss_mean_list=list(checkpoint.get('loss_mean_list', [])),
        best_val=float(checkpoint.get('best_val', float('-inf'))),
        best_test=float(checkpoint.get('best_test', float('-inf'))),
        checkpoint_path=resume_path,
    )


def build_checkpoint_payload(args, model, optimizer, lr_scheduler, epoch: int, loss_mean_list, window_state_version: int,
                             best_val: float, best_test: float) -> dict[str, Any]:
    return {
        'epoch': epoch,
        'start_epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'args_snapshot': {key: getattr(args, key) for key in _CHECKPOINT_COMPAT_KEYS},
        'loss_mean_list': list(loss_mean_list),
        'window_state_version': int(window_state_version),
        'best_val': float(best_val),
        'best_test': float(best_test),
    }


def save_training_checkpoint(args, payload: dict[str, Any], epoch: int, is_best: bool = False) -> list[str]:
    saved_paths: list[str] = []
    ensure_checkpoint_dir(args)
    paths = checkpoint_paths(args, epoch)

    torch.save(payload, paths['last'])
    saved_paths.append(str(paths['last']))

    save_every = max(int(args.save_checkpoint_every), 0)
    if save_every > 0 and (epoch + 1) % save_every == 0:
        torch.save(payload, paths['epoch'])
        saved_paths.append(str(paths['epoch']))

    if is_best:
        torch.save(payload, paths['best'])
        saved_paths.append(str(paths['best']))

    return saved_paths
