#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import random
from contextlib import nullcontext
from typing import Dict, Iterable, Iterator, List, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, Sampler, SequentialSampler, Subset
from tqdm import tqdm

from dataset import CityDataset, city_collate, infer_feature_dims, parse_county_ids
from model import UrbanMotifModel, urbanmotif_loss
from train_utils import (
    autocast_context,
    cleanup_distributed,
    load_checkpoint,
    save_checkpoint,
    setup_distributed,
    unwrap_model,
)
from utils import is_main_process, load_yaml, save_yaml, set_seed, setup_torch_performance


class NodeBudgetBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler: Sampler[int] | Iterable[int],
        *,
        num_nodes_by_index: List[int],
        max_nodes_per_batch: int,
        max_items_per_batch: int = 0,
        sort_pool_size: int = 0,
        pad_for_ddp: bool = True,
    ):
        self.sampler = sampler
        self.num_nodes_by_index = [int(x) for x in num_nodes_by_index]
        self.max_nodes_per_batch = int(max_nodes_per_batch)
        self.max_items_per_batch = int(max_items_per_batch)
        self.sort_pool_size = int(sort_pool_size)
        self.pad_for_ddp = bool(pad_for_ddp)
        if self.max_nodes_per_batch <= 0:
            raise ValueError("max_nodes_per_batch must be > 0")

    def _ordered_indices(self) -> List[int]:
        indices = [int(i) for i in self.sampler]
        if self.sort_pool_size > 1:
            ordered: List[int] = []
            for start in range(0, len(indices), self.sort_pool_size):
                chunk = indices[start : start + self.sort_pool_size]
                chunk.sort(key=lambda idx: self.num_nodes_by_index[idx], reverse=True)
                ordered.extend(chunk)
            indices = ordered
        return indices

    def _build_batches(self, indices: List[int]) -> List[List[int]]:
        batches: List[List[int]] = []
        batch: List[int] = []
        batch_nodes = 0
        for idx in indices:
            city_nodes = max(1, int(self.num_nodes_by_index[idx]))
            hit_item_limit = self.max_items_per_batch > 0 and len(batch) >= self.max_items_per_batch
            hit_node_limit = batch and (batch_nodes + city_nodes > self.max_nodes_per_batch)
            if hit_item_limit or hit_node_limit:
                batches.append(batch)
                batch = []
                batch_nodes = 0
            batch.append(idx)
            batch_nodes += city_nodes
        if batch:
            batches.append(batch)
        return batches

    def _ddp_max_batches(self, local_batches: int) -> int:
        if not self.pad_for_ddp or not dist.is_available() or not dist.is_initialized():
            return int(local_batches)
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        value = torch.tensor([int(local_batches)], device=device, dtype=torch.int64)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return int(value.item())

    def _pad_batches_for_ddp(self, batches: List[List[int]]) -> List[List[int]]:
        target = self._ddp_max_batches(len(batches))
        if len(batches) >= target or not batches:
            return batches
        filler = list(batches[-1])
        return batches + [list(filler) for _ in range(target - len(batches))]

    def __iter__(self) -> Iterator[List[int]]:
        indices = self._ordered_indices()
        batches = self._pad_batches_for_ddp(self._build_batches(indices))
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        local = len(self._build_batches(self._ordered_indices()))
        return self._ddp_max_batches(local)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--counties", default="", help="Optional comma/space-separated county FIPS filter.")
    p.add_argument("--county-fips-file", default="", help="Optional CSV/list with a county_fips column.")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    p.add_argument("--early-stopping-warmup", type=int, default=0)
    p.add_argument("--num-motifs", type=int, default=64)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--smooth-steps", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument(
        "--view-smooth-type",
        choices=["learnable", "learnable_diffuse", "learnable_diffusion", "diffuse"],
        default="learnable_diffuse",
    )
    p.add_argument(
        "--tract-context-type",
        choices=[
            "none",
            "identity",
            "no",
            "gcn",
            "gnn",
            "sparse_gcn",
            "sparse_gnn",
        ],
        default="gcn",
    )
    p.add_argument(
        "--tract-context-position",
        choices=["none", "off", "post", "after", "after_residual", "post_residual"],
        default="post_residual",
    )
    p.add_argument("--tract-context-graph", choices=["spatial", "space"], default="spatial")
    p.add_argument("--tract-context-layers", type=int, default=2)
    p.add_argument(
        "--residual-fusion",
        choices=[
            "attn",
            "attention",
            "attn_gated",
            "attention_gated",
            "resattn",
            "resattn_gated",
            "attn_add",
            "attention_add",
            "resattn_add",
            "none",
            "off",
        ],
        default="attn_gated",
    )
    p.add_argument("--motif-consensus-type", choices=["shared", "shared_motif", "view_mean", "mean", "no_motif", "none", "off"], default="shared")
    p.add_argument(
        "--profile-residual-decomp",
        choices=["on", "off", "true", "false", "yes", "no", "1", "0", "raw", "raw_views"],
        default="on",
        help="Set to off/raw to train directly from raw view encodings.",
    )
    p.add_argument("--enabled-views", default="all", help="Input views to keep, e.g. all, poi,lu,source,destination, wo_poi, wo_landuse, wo_mobility, only_poi.")
    p.add_argument("--poi-loss-weight", type=float, default=0.005)
    p.add_argument("--landuse-loss-weight", type=float, default=0.005)
    p.add_argument("--mobility-recon-loss-weight", type=float, default=0.015)
    p.add_argument("--contrast-loss-weight", type=float, default=0.1)
    p.add_argument("--contrast-temperature", type=float, default=0.5)
    p.add_argument("--graph-contrast-loss-weight", type=float, default=0.03)
    p.add_argument("--graph-contrast-temperature", type=float, default=0.2)
    p.add_argument("--graph-recon-temperature", type=float, default=0.2)
    p.add_argument("--pairwise-loss-sample-size", type=int, default=1024)
    p.add_argument("--max-abs-embedding", type=float, default=20.0)
    p.add_argument("--logit-clip", type=float, default=30.0)
    p.add_argument("--balance-weight", type=float, default=0.003)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--local-epochs-per-batch", type=int, default=1)
    p.add_argument("--cities-per-gpu", type=int, default=0)
    p.add_argument("--max-nodes-per-gpu", type=int, default=12000, help="If >0, greedily pack cities up to this many tract nodes per GPU batch instead of using a fixed city count.")
    p.add_argument("--batch-sort-pool-size", type=int, default=32, help="If >1, locally sort shuffled city indices by num_nodes within windows of this size before packing node-budget batches.")
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--dataset-cache", choices=["off", "cpu"], default="cpu")
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--find-unused-parameters", action="store_true", help="Enable DDP unused-parameter detection.")
    p.add_argument("--resume", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def parse_args():
    return build_arg_parser().parse_args()


def validate_args(args) -> None:
    for name in ("grad_accum_steps", "local_epochs_per_batch", "eval_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1")
    for name in (
        "num_workers",
        "cities_per_gpu",
        "warmup_epochs",
        "tract_context_layers",
        "poi_loss_weight",
        "landuse_loss_weight",
        "mobility_recon_loss_weight",
        "contrast_loss_weight",
        "balance_weight",
        "graph_contrast_loss_weight",
        "graph_contrast_temperature",
        "graph_recon_temperature",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "early_stopping_warmup",
        "pairwise_loss_sample_size",
        "max_abs_embedding",
        "logit_clip",
        "max_nodes_per_gpu",
        "batch_sort_pool_size",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.max_nodes_per_gpu == 0 and args.cities_per_gpu < 1:
        raise ValueError("--cities-per-gpu must be >= 1 when --max-nodes-per-gpu is disabled")
    if args.min_lr_ratio <= 0:
        raise ValueError("--min-lr-ratio must be > 0")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must satisfy 0 < ratio < 1")


def lr_scale(epoch: int, epochs: int, warmup_epochs: int, min_lr_ratio: float) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    if min_lr_ratio >= 1.0:
        return 1.0
    span = max(1, epochs - warmup_epochs - 1)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / span))
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def city_ids_from_paths(paths: Sequence[str]) -> List[str]:
    return [os.path.splitext(os.path.basename(path))[0] for path in paths]


def split_train_val_indices(
    city_ids: Sequence[str],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[List[int], List[int], Dict[str, object]]:
    total = len(city_ids)
    all_indices = list(range(total))
    if total < 2:
        raise ValueError("Pretraining needs at least two city graphs for a train/validation split.")
    val_count = max(1, int(round(total * float(val_ratio))))
    val_count = min(val_count, total - 1)
    rng = random.Random(int(seed))
    shuffled = list(all_indices)
    rng.shuffle(shuffled)
    val_set = set(shuffled[:val_count])
    train_indices = [idx for idx in all_indices if idx not in val_set]
    val_indices = [idx for idx in all_indices if idx in val_set]
    return train_indices, val_indices, {
        "mode": "fixed_random_city_split",
        "requested_val_ratio": float(val_ratio),
        "num_total_cities": int(total),
        "num_train_cities": int(len(train_indices)),
        "num_val_cities": int(len(val_indices)),
        "train_city_ids": [city_ids[idx] for idx in train_indices],
        "val_city_ids": [city_ids[idx] for idx in val_indices],
    }


def subset_num_nodes(dataset: CityDataset, indices: Sequence[int]) -> List[int]:
    return [dataset.get_num_nodes(int(idx)) for idx in indices]


def make_city_loader(
    dataset,
    *,
    num_nodes_by_index: Sequence[int],
    device: torch.device,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    distributed: bool,
    shuffle: bool,
    cities_per_gpu: int,
    max_nodes_per_gpu: int,
    batch_sort_pool_size: int,
    pad_for_ddp: bool,
):
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    batch_sampler = None
    loader_kwargs = {
        "num_workers": num_workers,
        "collate_fn": city_collate,
        "pin_memory": device.type == "cuda" and pin_memory,
        "persistent_workers": num_workers > 0 and persistent_workers,
    }
    if max_nodes_per_gpu > 0:
        base_sampler: Sampler[int] | Iterable[int]
        base_sampler = sampler if sampler is not None else SequentialSampler(dataset)
        batch_sampler = NodeBudgetBatchSampler(
            base_sampler,
            num_nodes_by_index=list(num_nodes_by_index),
            max_nodes_per_batch=max_nodes_per_gpu,
            max_items_per_batch=max(0, int(cities_per_gpu)),
            sort_pool_size=int(batch_sort_pool_size) if shuffle else 0,
            pad_for_ddp=pad_for_ddp,
        )
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs["batch_size"] = max(1, int(cities_per_gpu))
        loader_kwargs["sampler"] = sampler
        loader_kwargs["shuffle"] = bool(shuffle and sampler is None)
    if num_workers > 0:
        effective_prefetch = max(1, int(prefetch_factor))
        if batch_sampler is not None:
            effective_prefetch = 1
        loader_kwargs["prefetch_factor"] = effective_prefetch
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, sampler


def compute_city_loss(
    city,
    model,
    core_model,
    device,
    amp: str,
    args,
):
    city = city.to(device)
    with autocast_context(device, amp):
        out = model(city)
        loss, parts = urbanmotif_loss(
            out,
            city,
            sample_size=args.pairwise_loss_sample_size,
            contrast_temperature=args.contrast_temperature,
            poi_weight=args.poi_loss_weight,
            lu_weight=args.landuse_loss_weight,
            mobility_recon_weight=args.mobility_recon_loss_weight,
            contrast_weight=args.contrast_loss_weight,
            balance_weight=args.balance_weight,
            graph_contrast_weight=args.graph_contrast_loss_weight,
            graph_contrast_temperature=args.graph_contrast_temperature,
            graph_recon_temperature=args.graph_recon_temperature,
            max_abs_embedding=args.max_abs_embedding,
            logit_clip=args.logit_clip,
        )
    return loss, parts


def save_epoch_checkpoints(
    path: str,
    best_path: str,
    *,
    epoch: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    best_epoch: int,
    early_stop_bad_epochs: int,
    model,
    optimizer,
    scaler,
    improved: bool,
):
    extra = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": int(best_epoch),
        "early_stop_bad_epochs": int(early_stop_bad_epochs),
    }
    if improved:
        save_checkpoint(best_path, model, optimizer, scaler, epoch, extra=extra)
    save_checkpoint(path, model, optimizer, scaler, epoch, extra=extra)


def compute_batch_loss(
    cities,
    model,
    core_model,
    device,
    amp: str,
    args,
    *,
    return_parts: bool = False,
):
    batch_loss = torch.zeros((), device=device)
    part_sums: Dict[str, float] = {}
    for city in cities:
        loss, parts = compute_city_loss(
            city,
            model,
            core_model,
            device,
            amp,
            args,
        )
        batch_loss = batch_loss + loss
        if return_parts:
            for key, value in parts.items():
                part_sums[key] = part_sums.get(key, 0.0) + float(value)
    mean_loss = batch_loss / len(cities)
    if not return_parts:
        return mean_loss, len(cities)
    part_means = {key: value / len(cities) for key, value in part_sums.items()}
    return mean_loss, len(cities), part_means


@torch.no_grad()
def evaluate_loader(
    loader,
    model,
    core_model,
    device,
    amp: str,
    args,
):
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    total_count = 0
    part_sums: Dict[str, float] = {}
    try:
        for cities in loader:
            batch_loss, city_count, parts = compute_batch_loss(
                cities,
                model,
                core_model,
                device,
                amp,
                args,
                return_parts=True,
            )
            total_loss += float(batch_loss.detach().cpu()) * city_count
            total_count += city_count
            for key, value in parts.items():
                part_sums[key] = part_sums.get(key, 0.0) + float(value) * city_count
    finally:
        if model_was_training:
            model.train()
    denom = max(1, total_count)
    mean_loss = total_loss / denom
    mean_parts = {key: value / denom for key, value in part_sums.items()}
    mean_parts.setdefault("total_loss", mean_loss)
    return mean_loss, mean_parts


def main():
    args = parse_args()
    validate_args(args)
    setup_torch_performance()
    device, rank, world_size, local_rank = setup_distributed()
    try:
        main_process = is_main_process()
        set_seed(args.seed + rank)
        if main_process:
            save_yaml(vars(args), os.path.join(args.out_dir, "args.yaml"))

        poi_dim, lu_dim, source_dim, destination_dim = infer_feature_dims(args.processed_root)
        model = UrbanMotifModel(
            poi_dim=poi_dim,
            lu_dim=lu_dim,
            source_dim=source_dim,
            destination_dim=destination_dim,
            dim=args.dim,
            num_motifs=args.num_motifs,
            smooth_steps=args.smooth_steps,
            temperature=args.temperature,
            view_smooth_type=args.view_smooth_type,
            tract_context_type=args.tract_context_type,
            tract_context_position=args.tract_context_position,
            tract_context_graph=args.tract_context_graph,
            tract_context_layers=args.tract_context_layers,
            residual_fusion=args.residual_fusion,
            motif_consensus_type=args.motif_consensus_type,
            profile_residual_decomp=args.profile_residual_decomp,
            enabled_views=args.enabled_views,
            dropout=args.dropout,
        ).to(device)

        if args.compile:
            model = torch.compile(model)

        if world_size > 1:
            ddp_kwargs = {
                "device_ids": [local_rank],
                "output_device": local_rank,
                "broadcast_buffers": False,
                "gradient_as_bucket_view": True,
                "bucket_cap_mb": 64,
                "find_unused_parameters": bool(args.find_unused_parameters),
            }
            model = DDP(model, **ddp_kwargs)
        core_model = unwrap_model(model)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp == "fp16"))

        optimizer = torch.optim.AdamW(
            (p for p in core_model.parameters() if p.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        start_epoch = 0
        if args.resume:
            ckpt = load_checkpoint(args.resume, model, optimizer, scaler=scaler)
            extra = ckpt.get("extra") or {}
            start_epoch = int(ckpt.get("epoch", 0)) + 1
        ds = CityDataset(
            args.processed_root,
            cache_mode=args.dataset_cache,
            county_ids=parse_county_ids(args.counties),
            county_fips_file=args.county_fips_file,
        )
        split_path = os.path.join(args.out_dir, "val_split.yaml")
        history_path = os.path.join(args.out_dir, "training_history.yaml")
        city_ids = city_ids_from_paths(ds.paths)
        train_indices, val_indices, split_summary = split_train_val_indices(
            city_ids,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        if main_process:
            save_yaml(split_summary, split_path)
        train_ds = Subset(ds, train_indices)
        val_ds = Subset(ds, val_indices)
        train_num_nodes = subset_num_nodes(ds, train_indices)
        val_num_nodes = subset_num_nodes(ds, val_indices)
        loader, sampler = make_city_loader(
            train_ds,
            num_nodes_by_index=train_num_nodes,
            device=device,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory,
            distributed=world_size > 1,
            shuffle=True,
            cities_per_gpu=args.cities_per_gpu,
            max_nodes_per_gpu=args.max_nodes_per_gpu,
            batch_sort_pool_size=args.batch_sort_pool_size,
            pad_for_ddp=world_size > 1,
        )
        val_loader = None
        if main_process:
            val_loader, _ = make_city_loader(
                val_ds,
                num_nodes_by_index=val_num_nodes,
                device=device,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                persistent_workers=args.persistent_workers,
                pin_memory=args.pin_memory,
                distributed=False,
                shuffle=False,
                cities_per_gpu=args.cities_per_gpu,
                max_nodes_per_gpu=args.max_nodes_per_gpu,
                batch_sort_pool_size=args.batch_sort_pool_size,
                pad_for_ddp=False,
            )
        best_loss = float("inf")
        best_epoch = -1
        early_stop_bad_epochs = 0
        history: List[Dict[str, object]] = []
        if args.resume and os.path.exists(history_path):
            existing_history = load_yaml(history_path)
            if isinstance(existing_history, dict) and isinstance(existing_history.get("history"), list):
                history = list(existing_history["history"])
        if args.resume:
            best_loss = float(extra.get("best_val_loss", float("inf")))
            best_epoch = int(extra.get("best_epoch", -1))
            early_stop_bad_epochs = int(extra.get("early_stop_bad_epochs", 0))
        if main_process:
            if args.dataset_cache == "cpu" and args.num_workers > 0:
                print(
                    "[pretrain] dataset_cache=cpu with num_workers>0 caches samples independently in each worker; "
                    "this is fast but uses more host RAM.",
                    flush=True,
                )
            batching_desc = (
                f"node_budget={args.max_nodes_per_gpu}, city_cap={args.cities_per_gpu}, sort_pool={args.batch_sort_pool_size}"
                if args.max_nodes_per_gpu > 0
                else f"fixed_cities_per_gpu={args.cities_per_gpu}"
            )
            print(
                f"[pretrain] world_size={world_size}, dataset_cache={args.dataset_cache}, batching={batching_desc}",
                flush=True,
            )
            print(
                f"[pretrain] train_cities={len(train_indices)}, val_cities={len(val_indices)}, "
                f"val_mode={split_summary['mode']}, checkpoint=lowest_val_loss, "
                f"local_epochs_per_batch={args.local_epochs_per_batch}",
                flush=True,
            )

        optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, args.epochs):
            epoch_lr = args.lr * lr_scale(epoch, args.epochs, args.warmup_epochs, args.min_lr_ratio)
            set_optimizer_lr(optimizer, epoch_lr)
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            total_loss = torch.tensor(0.0, device=device)
            total_count = torch.tensor(0.0, device=device)
            steps_per_epoch = len(loader)
            progress = tqdm(loader, desc=f"epoch {epoch}", total=steps_per_epoch) if main_process else loader
            window_batches = []
            for step, cities in enumerate(progress):
                window_batches.append(cities)
                should_flush = len(window_batches) >= args.grad_accum_steps or (step + 1) == steps_per_epoch
                if not should_flush:
                    continue
                window_city_count = sum(len(batch) for batch in window_batches)
                last_batch_loss = None
                for _ in range(int(args.local_epochs_per_batch)):
                    for micro_idx, micro_cities in enumerate(window_batches):
                        should_step = (micro_idx + 1) == len(window_batches)
                        sync_context = (
                            nullcontext()
                            if should_step or not isinstance(model, DDP)
                            else model.no_sync()
                        )
                        with sync_context:
                            batch_loss, cities_in_batch = compute_batch_loss(
                                micro_cities,
                                model,
                                core_model,
                                device,
                                args.amp,
                                args,
                            )
                            scale = float(cities_in_batch) / max(1, int(window_city_count))
                            scaler.scale(batch_loss * scale).backward()
                        total_loss += batch_loss.detach() * cities_in_batch
                        total_count += cities_in_batch
                        last_batch_loss = batch_loss
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(core_model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                if main_process and last_batch_loss is not None:
                    progress.set_postfix(
                        loss=float(last_batch_loss.detach().cpu()),
                        accum=len(window_batches),
                        cities=window_city_count,
                        local=int(args.local_epochs_per_batch),
                    )
                window_batches.clear()

            if world_size > 1:
                dist.all_reduce(total_loss)
                dist.all_reduce(total_count)
            epoch_loss = (total_loss / total_count).item()
            val_loss = float("nan")
            val_parts: Dict[str, float] = {}
            improved = False
            early_stop_triggered = False
            should_eval = ((epoch + 1) % args.eval_every == 0) or (epoch + 1 == args.epochs)
            if main_process:
                if should_eval:
                    val_loss, val_parts = evaluate_loader(
                        val_loader,
                        core_model,
                        core_model,
                        device,
                        args.amp,
                        args,
                    )
                    improved = math.isfinite(val_loss) and val_loss < (best_loss - float(args.early_stopping_min_delta))
                    if improved:
                        best_loss = float(val_loss)
                        best_epoch = int(epoch)
                        early_stop_bad_epochs = 0
                    elif args.early_stopping_patience > 0:
                        if (epoch + 1) > args.early_stopping_warmup:
                            early_stop_bad_epochs += 1
                        else:
                            early_stop_bad_epochs = 0
                    if (
                        args.early_stopping_patience > 0
                        and (epoch + 1) > args.early_stopping_warmup
                        and early_stop_bad_epochs >= args.early_stopping_patience
                    ):
                        early_stop_triggered = True
                print(
                    f"epoch={epoch} train_loss={epoch_loss:.6f} "
                    f"val_loss={(val_loss if math.isfinite(val_loss) else float('nan')):.6f} "
                    f"best_val_loss={(best_loss if math.isfinite(best_loss) else float('nan')):.6f} "
                    f"lr={epoch_lr:.8g}",
                    flush=True,
                )
                save_epoch_checkpoints(
                    os.path.join(args.out_dir, "last.pt"),
                    os.path.join(args.out_dir, "best.pt"),
                    epoch=epoch,
                    train_loss=epoch_loss,
                    val_loss=val_loss,
                    best_val_loss=best_loss,
                    best_epoch=best_epoch,
                    early_stop_bad_epochs=early_stop_bad_epochs,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    improved=improved,
                )
                history.append(
                    {
                        "epoch": int(epoch),
                        "lr": float(epoch_lr),
                        "train_loss": float(epoch_loss),
                        "val_loss": float(val_loss) if math.isfinite(val_loss) else None,
                        "val_parts": {k: float(v) for k, v in val_parts.items()},
                        "best_val_loss": float(best_loss) if math.isfinite(best_loss) else None,
                        "best_epoch": int(best_epoch),
                        "improved": bool(improved),
                        "early_stop_bad_epochs": int(early_stop_bad_epochs),
                        "early_stopped": bool(early_stop_triggered),
                    }
                )
                save_yaml(
                    {
                        "split_summary": split_summary,
                        "history": history,
                    },
                    history_path,
                )
            if world_size > 1:
                dist.barrier()
                stop_flag = torch.tensor([1 if early_stop_triggered else 0], device=device, dtype=torch.int32)
                dist.broadcast(stop_flag, src=0)
                early_stop_triggered = bool(stop_flag.item())
            if early_stop_triggered:
                if main_process:
                    best_epoch_str = best_epoch if best_epoch >= 0 else None
                    print(
                        f"[pretrain] early stopping at epoch={epoch} best_epoch={best_epoch_str} "
                        f"best_val_loss={(best_loss if math.isfinite(best_loss) else float('nan')):.6f}",
                        flush=True,
                    )
                break
        if main_process and best_epoch < 0:
            raise RuntimeError("Training finished without a finite validation loss; best.pt was not written.")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
