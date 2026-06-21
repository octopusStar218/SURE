from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> tuple[torch.device, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    while True:
        if isinstance(model, DDP):
            model = model.module
            continue
        orig_mod = getattr(model, "_orig_mod", None)
        if orig_mod is not None:
            model = orig_mod
            continue
        return model


def autocast_context(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def save_checkpoint(path: str, model, optimizer=None, scaler=None, epoch: int = 0, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {
        "model": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        obj["scaler"] = scaler.state_dict()
    torch.save(obj, path)


def load_checkpoint(path: str, model, optimizer=None, scaler=None, strict: bool = True):
    ckpt = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(ckpt["model"], strict=strict)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt
