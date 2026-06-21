from __future__ import annotations

from typing import Tuple

import torch

torch.sparse.check_sparse_tensor_invariants.disable()


def torch_sparse_from_saved(indices: torch.Tensor, values: torch.Tensor, size: Tuple[int, int], device=None) -> torch.Tensor:
    t = torch.sparse_coo_tensor(indices.long(), values.float(), size=size).coalesce()
    if device is not None:
        t = t.to(device)
    return t


def torch_sparse_mm(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type=a.device.type, enabled=False):
        return torch.sparse.mm(a.float(), x.float())
